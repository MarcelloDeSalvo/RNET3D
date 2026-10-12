"""One training implementation for fixed splits and reusable K-Fold factories."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .data import PreparedData, SubjectSplit
from ..evaluation.k_fold import (
    EvaluationConfig, ExperimentMetadata, FoldContext, FoldSplit, TrainingResult,
    k_fold_loop, save_training_result, train_fold,
)


def configure_runtime(
    *, seed: int = 42, memory_growth: bool = True, mixed_precision: bool = False,
) -> dict[str, Any]:
    """Configure TensorFlow before creating models or generators.

    Seeding does not guarantee bitwise reproducibility on every accelerator.
    Changing memory growth after a device is initialized raises a clear error.
    """
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, memory_growth)

    policy: str = "float32"
    if mixed_precision:
        policy = "mixed_float16"
    tf.keras.mixed_precision.set_global_policy(policy)
    tf.keras.utils.set_random_seed(seed)

    return {"tensorflow": tf.__version__, "gpu_devices": [device.name for device in devices],
            "precision": policy, "seed": seed}


def build_model(config: ExperimentConfig) -> Any:
    """Build and compile the existing architecture with explicit head semantics."""
    import tensorflow as tf
    from ..evaluation.losses import diceBCELoss, diceCELoss, diceCELoss_gradient_fix, weightedBinaryCrossentropy
    from ..evaluation.metrics import dice_coefficient
    from ..models.networks import mt_r_net_3d, standard_unet_3d

    settings = config.model
    residual: bool = settings.architecture == "3drnet"
    attention: bool = settings.architecture == "3drnet"
    if settings.residual is not None:
        residual = settings.residual
    if settings.attention is not None:
        attention = settings.attention

    parameters = {
        "in_shape": config.preprocessing.patch_size,
        "in_channels": 1,
        "num_classes": len(config.labels.names),
        "filters": settings.filters,
        "residual": residual,
        "attention": attention,
        "max_pooling": settings.max_pooling,
    }
    if settings.architecture == "3drnet":
        parameters["with_classifier"] = settings.classifier
        model = mt_r_net_3d(**parameters)
    else:
        model = standard_unet_3d(**parameters)

    loss_settings = config.loss
    loss_parameters = {
        "gamma": loss_settings.gamma, "smooth": loss_settings.smooth,
        "batch_wise": loss_settings.batch_wise,
    }
    roi_head: str = "regions"
    metrics: dict[str, list[Any]] = {}
    if loss_settings.regions == "dice_ce_gradient_fix":
        roi_loss = diceCELoss_gradient_fix(**loss_parameters)
        logits = model.get_layer("regions").input
        roi_head = "regions_logits"
        logits_output = tf.keras.layers.Activation("linear", dtype="float32", name=roi_head)(logits)
        outputs = [logits_output, *model.outputs[1:]]
        model = tf.keras.Model(inputs=model.inputs, outputs=outputs, name="training_logits")
        # Historical probability metrics are not valid directly on this head.
    else:
        roi_loss = diceCELoss(**loss_parameters)
        roi_dice = dice_coefficient(mode="soft")
        metrics[roi_head] = [roi_dice]

    brain_loss = diceBCELoss(
        gamma=loss_settings.brain_gamma, alpha=loss_settings.brain_alpha,
        smooth=loss_settings.smooth, batch_wise=loss_settings.batch_wise,
    )
    brain_dice = dice_coefficient(exclude_background=False)
    losses = {roi_head: roi_loss, "brain_mask": brain_loss}
    metrics["brain_mask"] = [brain_dice]
    if settings.classifier:
        losses["classifier"] = weightedBinaryCrossentropy()

    optimizer = tf.keras.optimizers.Nadam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    return model


def inference_model(model: Any) -> Any:
    """Expose full softmax probabilities when a training view emits ROI logits."""
    import tensorflow as tf

    if model.output_names[0] == "regions":
        return model
    if model.output_names[0] != "regions_logits":
        raise ValueError("Expected regions probabilities or regions_logits")

    probabilities = tf.keras.layers.Activation("softmax", dtype="float32", name="regions")(model.outputs[0])
    outputs = [probabilities, *model.outputs[1:]]
    return tf.keras.Model(inputs=model.inputs, outputs=outputs, name="inference_probabilities")


def export_model(model: Any, path: Path) -> None:
    """Save an inference graph without optimizer/loss serialization dependencies."""
    prediction_model = inference_model(model)
    prediction_model.save(str(path), include_optimizer=False)


def callbacks(config: ExperimentConfig, context: FoldContext) -> list[Any]:
    """Leave best-weight selection to the shared runner."""
    import tensorflow as tf

    return [tf.keras.callbacks.EarlyStopping(
        monitor=context.config.monitor, mode=context.config.monitor_mode,
        patience=config.early_stopping_patience, restore_best_weights=False,
    )]


def evaluation_factories(data: PreparedData) -> dict[str, Any]:
    """Build full-volume evaluator hooks usable by both fixed splits and K-Fold."""
    if not data.config.inference.enabled:
        return {}

    from ..evaluation.inference import ExperimentVolumeEvaluator

    def make_evaluator(model: Any, ids: list[str], options: dict[str, Any]) -> Any:
        """Evaluate probabilities even when the fitting graph exposes logits."""
        prediction_model = inference_model(model)
        return ExperimentVolumeEvaluator(model=prediction_model, data=data, subject_ids=ids, options=options)

    return {"regions": make_evaluator}


class ExperimentFactories:
    """Small adapters shared by fixed-split and K-Fold orchestration."""

    def __init__(self, data: PreparedData) -> None:
        """Keep one prepared dataset and its resolved experiment configuration."""
        self.data: PreparedData = data

    def training_data(self, ids: list[str]) -> Any:
        """Prepare augmented scans for the selected training animals."""
        return self.data.make_generator(ids, training=True)

    def validation_data(self, ids: list[str]) -> Any:
        """Prepare every validation scan without random preprocessing."""
        return self.data.make_generator(ids, training=False)

    def model(self) -> Any:
        """Build a fresh compiled architecture with the configured head semantics."""
        return build_model(self.data.config)

    def callbacks(self, context: FoldContext) -> list[Any]:
        """Provide fresh callbacks while the runner owns checkpoint selection."""
        return callbacks(self.data.config, context)


def save_provenance(config: ExperimentConfig, data: PreparedData, run_dir: Path) -> None:
    """Record resolved settings, identities, reference digest and environment."""
    import tensorflow as tf

    settings = config.to_dict()
    (run_dir / "experiment.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    manifest_path = run_dir / "subjects.csv"
    if not manifest_path.exists():
        manifest = data.subjects.manifest(data.subjects.subject_ids)
        manifest.to_csv(manifest_path, index=False)

    repo_root = Path(__file__).resolve().parents[2]
    revision: str | None = None
    dirty: bool | None = None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True)
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass

    reference_path = Path(config.preprocessing.reference)
    reference_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    metadata_path = run_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "configuration_sha256": config.fingerprint(), "git_revision": revision, "git_dirty": dirty,
        "python": sys.version.split()[0], "platform": platform.platform(), "tensorflow": tf.__version__,
        "precision": tf.keras.mixed_precision.global_policy().name,
        "devices": [device.name for device in tf.config.list_physical_devices()],
        "reference_sha256": reference_digest,
        "training_roi": config.loss.regions, "exported_roi": "softmax_probabilities",
        "metrics_grid": "reoriented_resampled_orthogonal_mm",
        "predictions_grid": "original_voxel_grid_with_output_class_ids",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_training(data: PreparedData, split: SubjectSplit) -> TrainingResult:
    """Fit a fresh fixed-split experiment using the same implementation as K-Fold.

    Test animals are evaluated only after restoring validation-selected weights.
    Existing output folders are never overwritten. Resume is not supported yet.
    """
    config = data.config
    config.validate()
    split.validate(data.subjects)
    settings = config.training
    count: int = len(split.train) + len(split.validation)
    settings.validate(count, cross_validation=False)

    run_dir = settings.start_run()
    split.save(run_dir / "splits.json")
    save_provenance(config, data, run_dir)
    folder = run_dir / "fold_0"
    folder.mkdir()
    fold = FoldSplit(index=0, train_ids=list(split.train), val_ids=list(split.validation))
    checkpoint = folder / "best.weights.h5"
    context = FoldContext(split=fold, output_dir=folder, config=settings, checkpoint_path=checkpoint)
    dataset_name = "+".join(dataset.name for dataset in config.data.datasets)
    metadata = ExperimentMetadata(
        experiment_name=settings.model_name, model_variant=config.model.architecture,
        dataset_name=dataset_name,
    )
    evaluators = evaluation_factories(data)
    result = TrainingResult(run_dir=run_dir, evaluations={name: [] for name in evaluators}, metadata=metadata)
    evaluation = EvaluationConfig(
        test_ids=split.test, evaluate_validation=False,
        save_predictions=config.inference.save_predictions,
    )

    factories = ExperimentFactories(data)
    train_fold(
        context=context, train_factory=factories.training_data, val_factory=factories.validation_data,
        model_factory=factories.model, callback_factory=factories.callbacks,
        evaluators=evaluators, evaluation_config=config.to_dict(), result=result,
        evaluation=evaluation, model_exporter=export_model,
    )
    save_training_result(result, settings)
    return result


def run_cross_validation(
    data: PreparedData, *, subject_ids: list[str] | None = None,
    precomputed_splits: Path | None = None, evaluation: EvaluationConfig | None = None,
) -> TrainingResult:
    """Connect the same factories to existing grouped cross-validation.

    Fold validation selects checkpoints and is diagnostic, not an independent
    outer test. Reserve a disjoint test cohort for final performance estimates.
    Configure CV splits separately from model initialization when comparing seeds.
    """
    config = data.config
    config.validate()
    if subject_ids is None:
        subject_ids = data.subjects.subject_ids
    if evaluation is None:
        evaluation = EvaluationConfig(save_predictions=config.inference.save_predictions)

    factories = ExperimentFactories(data)
    dataset_name = "+".join(dataset.name for dataset in config.data.datasets)
    metadata = ExperimentMetadata(
        experiment_name=config.training.model_name, model_variant=config.model.architecture,
        dataset_name=dataset_name,
    )

    def write_provenance(run_dir: Path) -> None:
        """Save complete settings even if a later training fold is interrupted."""
        save_provenance(config, data, run_dir)

    result = k_fold_loop(
        train_gen=factories.training_data, val_gen=factories.validation_data,
        model_callback=factories.model, config=config.training,
        train_and_test_ids=subject_ids, callbacks_factory=factories.callbacks,
        precomputed_splits=precomputed_splits, subject_index=data.subjects,
        evaluation=evaluation, metadata=metadata,
        fve_callbacks=evaluation_factories(data), model_exporter=export_model,
        split_seed=config.split_seed, run_setup=write_provenance,
    )
    return result


def history_figure(result: TrainingResult) -> Any:
    """Create a native Plotly figure without changing or retraining the experiment."""
    import plotly.graph_objects as go

    figure = go.Figure()
    for fold, history in enumerate(result.histories):
        for key in ("loss", "val_loss"):
            values = history[key]
            epochs = list(range(1, len(values) + 1))
            figure.add_trace(go.Scatter(x=epochs, y=values, mode="lines", name=f"Fold {fold}: {key}"))

    figure.update_layout(xaxis_title="Epoch", yaxis_title="Loss", title="Training history")
    return figure
