"""One training implementation for fixed splits and reusable K-Fold factories."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from copy import deepcopy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .data import PreparedData, SubjectSplit, make_cv_splits
from .initialization import initialize_model
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


def build_network(config: ExperimentConfig) -> Any:
    """Build the existing probability graph without transfer or compilation."""
    from ..models.networks import mt_r_net_3d, standard_unet_3d

    settings = config.model
    num_classes = len(config.labels.names)
    parameters = {
        "in_shape": config.preprocessing.patch_size,
        "in_channels": 1,
        "num_classes": num_classes,
        "filters": settings.filters,
        "residual": settings.residual,
        "attention": settings.attention,
        "max_pooling": settings.max_pooling,
    }
    if settings.architecture == "3drnet":
        parameters["with_classifier"] = settings.classifier
        model = mt_r_net_3d(**parameters)
    else:
        model = standard_unet_3d(**parameters)
    return model


def training_view(model: Any, config: ExperimentConfig) -> Any:
    """Expose the exact fitting outputs, including logits for the corrected loss."""
    import tensorflow as tf

    if config.loss.regions != "dice_ce_gradient_fix":
        return model
    logits = model.get_layer("regions").input
    logits_output = tf.keras.layers.Activation("linear", dtype="float32", name="regions_logits")(logits)
    outputs = [logits_output, *model.outputs[1:]]
    return tf.keras.Model(inputs=model.inputs, outputs=outputs, name="training_logits")


def build_model(config: ExperimentConfig) -> Any:
    """Build, initialize and compile the existing training architecture."""
    config.loss.validate_for_training()
    config.loss.validate_weights(with_classifier=config.model.classifier)
    import tensorflow as tf
    from ..evaluation.losses import diceBCELoss, diceCELoss, diceCELoss_gradient_fix, weightedBinaryCrossentropy
    from ..evaluation.metrics import PartialLabelDice, dice_coefficient

    model = build_network(config)
    initialization_report = initialize_model(model, config.initialization)
    model = training_view(model, config)
    settings = config.model
    num_classes = len(config.labels.names)

    loss_settings = config.loss
    loss_parameters = {
        "gamma": loss_settings.gamma, "smooth": loss_settings.smooth,
        "batch_wise": loss_settings.batch_wise,
    }
    roi_head: str = "regions"
    metrics: dict[str, list[Any]] = {}
    if loss_settings.regions == "dice_ce_gradient_fix":
        loss_parameters["version"] = loss_settings.gradient_fix_version
        if loss_settings.gradient_fix_version == 3:
            loss_parameters["positive_weight"] = loss_settings.foreground_positive_weight
        roi_loss = diceCELoss_gradient_fix(**loss_parameters)
        roi_head = "regions_logits"
        roi_dice = PartialLabelDice(num_classes=num_classes, from_logits=True)
        metrics[roi_head] = [roi_dice]
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
    task_weights = {roi_head: loss_settings.regions_weight, "brain_mask": loss_settings.brain_weight}
    metrics["brain_mask"] = [brain_dice]
    if settings.classifier:
        losses["classifier"] = weightedBinaryCrossentropy()
        task_weights["classifier"] = loss_settings.classifier_weight

    optimizer = tf.keras.optimizers.Nadam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss=losses, loss_weights=task_weights, metrics=metrics)
    model.initialization_report = initialization_report
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
    """Legacy fixed-split default; notebooks now pass their visible callback list."""
    import tensorflow as tf

    return [tf.keras.callbacks.EarlyStopping(
        monitor=context.config.monitor, mode=context.config.monitor_mode,
        patience=config.early_stopping_patience, restore_best_weights=False,
    )]


def callback_configuration(callback_list: Sequence[Any]) -> list[dict[str, Any]]:
    """Record standard Keras callback settings separately from changing fit state."""
    fields = (
        "monitor", "mode", "min_delta", "patience", "verbose", "baseline",
        "restore_best_weights", "factor", "cooldown", "min_lr", "log_dir",
        "histogram_freq", "write_graph", "write_images", "write_steps_per_second",
        "update_freq", "embeddings_freq", "embeddings_metadata", "filename", "sep", "append",
        "filepath", "save_best_only", "save_weights_only", "save_freq",
        "_start_batch", "_stop_batch",
    )
    records: list[dict[str, Any]] = []
    for callback in callback_list:
        settings = {name: getattr(callback, name) for name in fields if hasattr(callback, name)}
        if hasattr(callback, "monitor_op"):
            settings["monitor_direction"] = callback.monitor_op.__name__
        record = {"class": type(callback).__name__, "module": type(callback).__module__, "settings": settings}
        records.append(record)
    return json.loads(json.dumps(records, default=str))


def _callback_contract(callback_list: Sequence[Any] | None, folder: Path, *, resume: bool) -> None:
    """Save editable callback settings and reject changes during explicit-list resumption."""
    path = folder / "callbacks.json"
    if callback_list is None:
        if resume and path.exists():
            raise ValueError("Resume requires the saved explicit callback list")
        return

    configuration = callback_configuration(callback_list)
    if resume:
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved != configuration:
                raise ValueError("Resume callback settings differ from the saved run")
    else:
        path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")


def _copy_callbacks(callback_list: Sequence[Any], context: FoldContext) -> list[Any]:
    """Make fresh callbacks per fit and resolve relative logging paths in its fold."""
    import tensorflow as tf

    result: list[Any] = []
    for callback in callback_list:
        if not isinstance(callback, tf.keras.callbacks.Callback):
            raise TypeError("Pass Keras callback instances in callbacks")
        if callback.model is not None:
            raise ValueError("Pass unused callback instances; callback state is copied fresh for each fit")

        copied = deepcopy(callback)
        if isinstance(copied, tf.keras.callbacks.TensorBoard):
            log_dir = Path(copied.log_dir)
            if not log_dir.is_absolute():
                log_dir = context.output_dir / log_dir
            copied.log_dir = str(log_dir)
        if isinstance(copied, tf.keras.callbacks.CSVLogger):
            log_file = Path(copied.filename)
            if not log_file.is_absolute():
                log_file = context.output_dir / log_file
            copied.filename = str(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            recovery_file = context.output_dir / "training_state.json"
            if recovery_file.exists():
                copied.append = True
        if isinstance(copied, tf.keras.callbacks.ModelCheckpoint):
            checkpoint_file = Path(copied.filepath)
            if not checkpoint_file.is_absolute():
                checkpoint_file = context.output_dir / checkpoint_file
            copied.filepath = str(checkpoint_file)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        result.append(copied)
    return result


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

    def __init__(
        self, data: PreparedData, callback_list: Sequence[Any] | None = None,
        *, fixed_split: bool = True,
    ) -> None:
        """Keep one prepared dataset and its resolved experiment configuration."""
        self.data: PreparedData = data
        self.callback_list = callback_list
        self.fixed_split: bool = fixed_split
        if callback_list is not None and data.config.training.default_callbacks:
            raise ValueError("Set TrainingConfig.default_callbacks=False when providing an explicit callback list")

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
        """Copy notebook callbacks; never add implicit early stopping to CV."""
        if self.callback_list is not None:
            return _copy_callbacks(self.callback_list, context)
        if not self.fixed_split:
            return []
        return callbacks(self.data.config, context)


def save_provenance(config: ExperimentConfig, data: PreparedData, run_dir: Path) -> None:
    """Record resolved settings, identities, reference digest and environment."""
    import tensorflow as tf

    settings = config.to_dict()
    (run_dir / "experiment.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    contract = data.contract()
    (run_dir / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    data.save_evaluation_patches(run_dir / "evaluation_patches.json")
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
    initialization_digest: str | None = None
    if config.initialization.checkpoint is not None:
        checkpoint_path = Path(config.initialization.checkpoint)
        initialization_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    initialization_config_digest: str | None = None
    if config.initialization.source_experiment is not None:
        source_experiment = Path(config.initialization.source_experiment)
        initialization_config_digest = hashlib.sha256(source_experiment.read_bytes()).hexdigest()
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
        "initialization_sha256": initialization_digest,
        "initialization_config_sha256": initialization_config_digest,
        "training_roi": config.loss.regions, "exported_roi": "softmax_probabilities",
        "metrics_grid": "reoriented_resampled_orthogonal_mm",
        "predictions_grid": "original_voxel_grid_with_output_class_ids",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def validate_resume(data: PreparedData, run_dir: Path) -> None:
    """Reject changed experiments, input identities, annotations, geometry or precision."""
    import tensorflow as tf

    config = data.config
    saved = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    current = json.loads(json.dumps(config.to_dict(), default=str))
    if "fixed_split" not in saved:
        # Older runs saved memberships but not generation fractions. The runner
        # validates their split.json/splits.json separately; never regenerate it.
        saved["fixed_split"] = current["fixed_split"]
    saved = ExperimentConfig.from_dict(saved).to_dict()
    saved = json.loads(json.dumps(saved, default=str))
    if saved["loss"]["regions"] == "dice_ce_gradient_fix":
        if saved["loss"]["gradient_fix_version"] != config.loss.gradient_fix_version:
            raise ValueError("Partial-label loss objective changed; start a new run instead of resuming older weights")
    if saved != current:
        raise ValueError("Resume configuration differs from the saved experiment")
    contract = json.loads((run_dir / "data_contract.json").read_text(encoding="utf-8"))
    if contract != data.contract():
        raise ValueError("Resume data identities, annotations, geometry or source files have changed")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["precision"] != tf.keras.mixed_precision.global_policy().name:
        raise ValueError("Resume requires the saved precision policy")
    if metadata["tensorflow"] != tf.__version__:
        raise ValueError("Resume requires the saved TensorFlow version")
    reference = Path(config.preprocessing.reference)
    if metadata["reference_sha256"] != hashlib.sha256(reference.read_bytes()).hexdigest():
        raise ValueError("Resume reference image has changed")
    if config.initialization.checkpoint is not None:
        source = Path(config.initialization.checkpoint)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if metadata["initialization_sha256"] != digest:
            raise ValueError("Resume initialization checkpoint has changed")
    if config.initialization.source_experiment is not None:
        source_experiment = Path(config.initialization.source_experiment)
        source_digest = hashlib.sha256(source_experiment.read_bytes()).hexdigest()
        if metadata.get("initialization_config_sha256") != source_digest:
            raise ValueError("Resume initialization source experiment has changed")

    data.save_evaluation_patches(run_dir / "evaluation_patches.json", resume=True)


def run_training(
    data: PreparedData, split: SubjectSplit, *, resume_run: Path | None = None,
    callbacks: Sequence[Any] | None = None,
) -> TrainingResult:
    """Fit a fresh fixed-split experiment using the same implementation as K-Fold.

    Test animals are evaluated only after restoring validation-selected weights.
    Pass resume_run to recover completed epochs in an existing matching run.
    callbacks accepts the notebook's explicit list. Instances are copied for
    this fit; relative TensorBoard/CSV paths resolve inside fold_0. Disable
    TrainingConfig.default_callbacks to avoid adding another implicit list.
    """
    config = data.config
    config.validate()
    split.validate(data.subjects)
    settings = config.training
    count: int = len(split.train) + len(split.validation)
    settings.validate(count, cross_validation=False)
    factories = ExperimentFactories(data, callback_list=callbacks)

    resume: bool = resume_run is not None
    if resume:
        run_dir = Path(resume_run).resolve()
        validate_resume(data, run_dir)
        saved_split = SubjectSplit.load(run_dir / "splits.json", data.subjects)
        if split != saved_split:
            raise ValueError("Resume split differs from the saved experiment")
    else:
        run_dir = settings.start_run()
        split.save(run_dir / "splits.json")
        save_provenance(config, data, run_dir)
    _callback_contract(callbacks, run_dir, resume=resume)
    folder = run_dir / "fold_0"
    folder.mkdir(exist_ok=resume)
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

    from ..evaluation.training_state import run_lock

    with run_lock(run_dir):
        train_fold(
            context=context, train_factory=factories.training_data, val_factory=factories.validation_data,
            model_factory=factories.model, callback_factory=factories.callbacks,
            evaluators=evaluators, evaluation_config=config.to_dict(), result=result,
            evaluation=evaluation, model_exporter=export_model, resume=resume,
        )
        save_training_result(result, settings)
    return result


def run_cross_validation(
    data: PreparedData, *, subject_ids: list[str] | None = None,
    precomputed_splits: Path | None = None, evaluation: EvaluationConfig | None = None,
    resume_run: Path | None = None,
    folds: list[FoldSplit] | None = None,
    callbacks: Sequence[Any] | None = None,
) -> TrainingResult:
    """Connect the same factories to existing grouped cross-validation.

    By default each outer fold has separate training, selection-validation and
    holdout animals. Old two-role split files remain diagnostic selection folds.
    callbacks is copied separately for each fold; no early stopping is added
    implicitly. Relative log paths are resolved within each fold's directory.
    """
    config = data.config
    config.validate()
    if subject_ids is None:
        subject_ids = data.subjects.subject_ids
    if evaluation is None:
        evaluation = EvaluationConfig(evaluate_validation=False, save_predictions=config.inference.save_predictions)
    if resume_run is not None:
        validate_resume(data, Path(resume_run))
    elif folds is None and precomputed_splits is None:
        folds = make_cv_splits(
            subject_ids, k=config.training.k, seed=config.split_seed,
            validation_fraction=config.cv_validation_fraction,
        )

    factories = ExperimentFactories(data, callback_list=callbacks, fixed_split=False)
    dataset_name = "+".join(dataset.name for dataset in config.data.datasets)
    metadata = ExperimentMetadata(
        experiment_name=config.training.model_name, model_variant=config.model.architecture,
        dataset_name=dataset_name,
    )

    def write_provenance(run_dir: Path) -> None:
        """Save complete settings even if a later training fold is interrupted."""
        if resume_run is None:
            save_provenance(config, data, run_dir)
        _callback_contract(callbacks, run_dir, resume=resume_run is not None)

    result = k_fold_loop(
        train_gen=factories.training_data, val_gen=factories.validation_data,
        model_callback=factories.model, config=config.training,
        train_and_test_ids=subject_ids, callbacks_factory=factories.callbacks,
        precomputed_splits=precomputed_splits, subject_index=data.subjects,
        evaluation=evaluation, metadata=metadata,
        fve_callbacks=evaluation_factories(data), model_exporter=export_model,
        split_seed=config.split_seed, run_setup=write_provenance,
        resume_run=resume_run,
        fold_splits=folds,
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
