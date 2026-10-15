"""Reusable subject-level TensorFlow/Keras K-fold training.

Each dataset factory receives the subject IDs for its fold and must return a
fresh Keras-compatible dataset. For longitudinal data, provide one ID per
subject (or a custom grouped splitter), never one ID per repeated scan.

Example::

    config = TrainingConfig(model_name="segmentation", k=5, epochs=100)

    def extra_callbacks(context: FoldContext) -> list[tf.keras.callbacks.Callback]:
        return [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=False)]

    result = k_fold_loop(
        train_gen=make_train_dataset,
        val_gen=make_validation_dataset,
        model_callback=build_compiled_model,
        config=config,
        train_and_test_ids=subject_ids,
        callbacks_factory=extra_callbacks,
        fve_callbacks={"regions": RegionEvaluator},
    )

The runner always owns a best-weights checkpoint so that evaluation uses the
selected epoch. ``callbacks_factory`` can add callbacks or replace the default
TensorBoard, ReduceLROnPlateau, and CSVLogger callbacks.

Optional reviewer-ready setup::

    evaluation = EvaluationConfig(test_ids=("test_01",), external_ids=("d4_01",))
    metadata = ExperimentMetadata(model_variant="no_attention", dataset_name="D1")
    result = k_fold_loop(..., evaluation=evaluation, metadata=metadata,
                         precomputed_splits="previous_run/splits.json")

The external and test IDs must be disjoint from the CV subjects. Each selected
fold model is evaluated on each requested independent cohort; these repeat
measurements are **not** independent test observations. Evaluators should emit
one row per subject/metric, ideally with Subject, Metric, Value, and optionally
Region columns. The runner adds Fold, DatasetRole and Evaluator columns.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import re
import platform
import sys
import time
import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from ..utils.loader import RodentDatasets, SubjectKey

LOGGER = logging.getLogger(__name__)
History: TypeAlias = dict[str, list[float | None]]
DatasetFactory: TypeAlias = Callable[[list[str]], Any]
ModelFactory: TypeAlias = Callable[[], Any]
CallbackFactory: TypeAlias = Callable[["FoldContext"], Sequence[Any]]
SplitFactory: TypeAlias = Callable[[Sequence[str]], Iterable[tuple[Sequence[int], Sequence[int]]]]


class FullVolumeEvaluator(Protocol):
    """A full-volume evaluator exposing a DataFrame of subject metrics."""

    def evaluate(self) -> pd.DataFrame: ...


EvaluationFactory: TypeAlias = Callable[[Any, list[str], Mapping[str, Any]], FullVolumeEvaluator]
Evaluators: TypeAlias = Mapping[str, EvaluationFactory] | Sequence[EvaluationFactory]


@dataclass(frozen=True, kw_only=True)
class SegmentationVolume:
    """One reconstructed scan on a shared orthogonal grid, ready for evaluation.

    Reference and prediction are integer label maps, not probabilities/logits.
    Spacing is in millimetres in array-axis order. Caller must verify physical
    alignment, handle label mapping and provide annotation availability.
    """

    reference: np.ndarray
    prediction: np.ndarray
    spacing_mm: tuple[float, float, float]
    scan: str
    dataset: str = ''
    available_labels: Sequence[int] | None = None


VolumePredictor: TypeAlias = Callable[[Any, str], Iterable[SegmentationVolume]]


class SegmentationEvaluator:
    """Adapt full-volume predictions to the runner's long-form metric tables.

    The prediction callback receives the restored model and one canonical
    subject ID, then yields its reconstructed scans. It owns preprocessing,
    inference, reconstruction and any requested prediction-file saving.
    This class measures complete volumes and does not average timepoints.
    """

    def __init__(
        self, model: Any, subject_ids: Sequence[str], options: Mapping[str, Any], *,
        predict_subject: VolumePredictor, regions: Mapping[int, str],
    ) -> None:
        """Store the model, requested animals and explicit anatomical labels."""
        self.model: Any = model
        self.subject_ids: list[str] = list(subject_ids)
        self.options: dict[str, Any] = dict(options)
        self.predict_subject: VolumePredictor = predict_subject
        self.regions: dict[int, str] = dict(regions)

    def evaluate(self) -> pd.DataFrame:
        """Return one row per subject/scan/structure/metric, including statuses."""
        from .metrics import segmentation_metrics_table

        tables: list[pd.DataFrame] = []
        for subject in self.subject_ids:
            seen_scans: set[str] = set()
            volumes = self.predict_subject(self.model, subject)
            for volume in volumes:
                if not volume.scan or volume.scan in seen_scans:
                    raise ValueError(f'{subject}: scan identifiers must be nonempty and unique')
                seen_scans.add(volume.scan)
                table = segmentation_metrics_table(
                    volume.reference, volume.prediction, volume.spacing_mm, self.regions,
                    subject=subject, scan=volume.scan, dataset=volume.dataset,
                    available_labels=volume.available_labels,
                )
                tables.append(table)

            if not seen_scans:
                raise ValueError(f'No full-volume predictions returned for {subject}')

        if not tables:
            raise ValueError('No subjects supplied for full-volume evaluation')
        return pd.concat(tables, ignore_index=True)


class SubjectIndex:
    """Bridge subject-level folds to the loader's generator-compatible scan keys.

    Normalized timepoints share ``Rodent.subject_group``. Legacy scans retain
    the loader's modality/ID grouping. Explicit overrides can join identities
    confirmed to belong to the same animal; original IDs are never guessed.
    Build once after registering every CV/test/external dataset in the loader.
    """

    def __init__(
        self, loader: RodentDatasets, *,
        subject_overrides: Mapping[SubjectKey, str] | None = None,
    ) -> None:
        """Snapshot stable subject IDs, scan keys and source identity metadata."""
        overrides: dict[SubjectKey, str] = dict(subject_overrides or {})
        scan_keys: list[SubjectKey] = sorted(loader.get_subjects_list())
        unknown_keys = set(overrides) - set(scan_keys)
        if unknown_keys:
            raise ValueError(f"Subject overrides contain unknown scan keys: {unknown_keys}")

        self._scans: dict[str, list[SubjectKey]] = {}
        self._subjects: dict[SubjectKey, str] = {}
        self._records: list[dict[str, Any]] = []
        group_subjects: dict[SubjectKey, str] = {}

        for scan_key in scan_keys:
            scan = loader.get_subject(scan_key)
            group = scan.subject_group
            encoded_group: str = quote(group[0], safe="")
            encoded_subject: str = quote(group[1], safe="")
            subject_id: str = overrides.get(scan_key, f"{encoded_group}/{encoded_subject}")
            if not isinstance(subject_id, str) or not subject_id.strip():
                raise ValueError("Subject IDs must be nonempty strings")

            previous_id = group_subjects.setdefault(group, subject_id)
            if previous_id != subject_id:
                raise ValueError(f"Overrides must keep every scan of {group} together")

            self._scans.setdefault(subject_id, []).append(scan_key)
            self._subjects[scan_key] = subject_id
            self._records.append({
                "Subject": subject_id, "Group": scan_key[0], "Scan": scan_key[1],
                "Dataset": group[0], "SubjectID": scan.subject_id,
                "OriginalID": scan.original_id, "Timepoint": scan.timepoint,
                "Subgroup": scan.metadata.get("subgroup", ""),
                "Condition": scan.metadata.get("condition", "unknown"),
                "Species": scan.metadata.get("species", "unknown"),
                "Strain": scan.metadata.get("strain", "unknown"),
                "Protocol": scan.metadata.get("protocol", "unknown"),
                "Modality": scan.modality, "ImagePath": str(scan.image_path),
            })

    @property
    def subject_ids(self) -> list[str]:
        """Return sorted animal IDs for splitting, independently of scan order."""
        return sorted(self._scans)

    def scan_keys(self, subject_ids: Sequence[str]) -> list[SubjectKey]:
        """Expand selected animals to all their scans for a dataset factory."""
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("Subject selection contains duplicate IDs")

        keys: list[SubjectKey] = []
        for subject_id in subject_ids:
            if subject_id not in self._scans:
                raise ValueError(f"Unknown subject ID: {subject_id!r}")
            keys.extend(self._scans[subject_id])

        return keys

    def subject_for_scan(self, scan_key: SubjectKey) -> str:
        """Map an evaluation scan back to its canonical animal ID."""
        return self._subjects[scan_key]

    def manifest(self, subject_ids: Sequence[str]) -> pd.DataFrame:
        """Return one provenance row per selected scan without reading voxels."""
        self.scan_keys(subject_ids)
        selected: set[str] = set(subject_ids)
        records = [record for record in self._records if record["Subject"] in selected]
        return pd.DataFrame(records)


def _safe_name(value: str) -> str:
    """Validate a single output filename or folder component."""
    if not re.fullmatch(r"[\w.-]+", value) or value in {".", ".."}:
        raise ValueError(f"Invalid output identifier: {value!r}")
    return value


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """K-fold execution options.

    Attributes:
        model_name: Name of the experiment group.
        k: Expected number of folds, including with a custom splitter.
        epochs: Maximum epochs in each fold.
        batch_size: Batch size used only when inferring training steps.
        data_multiplier: Training-only multiplier when inferring steps.
        seed: Random seed used for splits and per-fold initialization.
        output_dir: Directory containing experiments.
        run_id: Optional unique run label; generated automatically if omitted.
        steps_per_epoch: Explicit training batches per epoch, or None.
        validation_steps: Explicit validation batches per epoch, or None.
        infer_steps_from_ids: Infer missing steps from the number of subjects.
            Keep False for datasets with their own cardinality or patch sampling.
        monitor: Validation metric used to select the best model weights.
        monitor_mode: Whether smaller or larger monitor values are better.
        default_callbacks: Whether to add standard logging and LR callbacks.
        histogram_freq: TensorBoard histogram frequency if enabled.
        save_model: Whether to save the restored model after training.
        fit_kwargs: Additional arguments passed to ``model.fit``. Do not pass
            its datasets, epochs, callbacks, or step arguments here.
    """

    model_name: str
    k: int = 5
    epochs: int = 100
    batch_size: int = 4
    data_multiplier: int = 1
    seed: int = 42
    output_dir: str = "../results"
    run_id: str | None = None
    steps_per_epoch: int | None = None
    validation_steps: int | None = None
    infer_steps_from_ids: bool = False
    monitor: str = "val_loss"
    monitor_mode: Literal["min", "max"] = "min"
    default_callbacks: bool = True
    histogram_freq: int = 0
    save_model: bool = True
    fit_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_input(cls, value: TrainingConfig | Mapping[str, Any]) -> TrainingConfig:
        """Normalize a dataclass or a legacy configuration dictionary."""
        if isinstance(value, cls):
            return value

        return cls(**value)

    def validate(self, n_subjects: int, *, cross_validation: bool = True) -> None:
        """Reject invalid configurations before starting an experiment."""
        _safe_name(self.model_name)
        if self.run_id is not None:
            _safe_name(self.run_id)
        if cross_validation and not 2 <= self.k <= n_subjects:
            raise ValueError("k must be between 2 and the number of unique subjects")
        for name in ("epochs", "batch_size", "data_multiplier"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("steps_per_epoch", "validation_steps"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        if self.monitor_mode not in {"min", "max"} or not self.monitor:
            raise ValueError("monitor_mode must be 'min' or 'max' and monitor cannot be empty")
        if self.histogram_freq < 0:
            raise ValueError("histogram_freq must be nonnegative")
        reserved = {"x", "validation_data", "epochs", "initial_epoch", "callbacks", "steps_per_epoch", "validation_steps"}
        if reserved.intersection(self.fit_kwargs):
            raise ValueError(f"fit_kwargs contains managed model.fit parameters: {reserved & self.fit_kwargs.keys()}")

    def start_run(self) -> Path:
        """Create an isolated run directory without overwriting old results."""
        run_id = self.run_id or datetime.now().strftime("RUN_%Y%m%d_%H%M%S_%f")
        folder = Path(self.output_dir) / self.model_name / _safe_name(run_id)
        folder.mkdir(parents=True, exist_ok=False)
        return folder

    def fit_steps(self, n_train: int, n_val: int) -> dict[str, int]:
        """Return requested or inferred batch counts for ``model.fit``."""
        steps: dict[str, int] = {}
        if self.steps_per_epoch is not None:
            steps["steps_per_epoch"] = self.steps_per_epoch
        elif self.infer_steps_from_ids:
            steps["steps_per_epoch"] = math.ceil(n_train * self.data_multiplier / self.batch_size)

        if self.validation_steps is not None:
            steps["validation_steps"] = self.validation_steps
        elif self.infer_steps_from_ids:
            steps["validation_steps"] = math.ceil(n_val / self.batch_size)
        return steps


@dataclass(frozen=True, kw_only=True)
class ExperimentMetadata:
    """Experiment provenance. Tags may describe architecture, cohort or ablation."""

    experiment_name: str = ""
    model_variant: str = ""
    dataset_name: str = ""
    dataset_version: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class EvaluationConfig:
    """Optional independent cohorts; these IDs must not occur in CV subjects.

    External evaluation is run once for *each* selected fold model. Results
    therefore describe multiple trained models, not independent test samples.
    """

    test_ids: tuple[str, ...] = ()
    external_ids: tuple[str, ...] = ()
    evaluate_validation: bool = True
    save_predictions: bool = False  # Instructs evaluator via its config; runner does not generate masks.

    def validate(self, cv_ids: Sequence[str]) -> None:
        """Reject duplicate subjects or overlap between evaluation cohorts."""
        for subject in (*self.test_ids, *self.external_ids):
            if not isinstance(subject, str) or not subject.strip():
                raise ValueError("Test and external subject IDs must be nonempty strings")

        cohorts = (set(cv_ids), set(self.test_ids), set(self.external_ids))
        if len(self.test_ids) != len(cohorts[1]) or len(self.external_ids) != len(cohorts[2]):
            raise ValueError("Test and external subject IDs must be unique")
        if any(a & b for i, a in enumerate(cohorts) for b in cohorts[i + 1:]):
            raise ValueError("CV, test and external subject IDs must be mutually disjoint")


@dataclass
class FoldTiming:
    """Elapsed wall-clock seconds and model complexity for one fold."""

    fold: int
    training_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    epochs_completed: int = 0
    parameter_count: int = 0
    best_epoch: int = 0


@dataclass(frozen=True)
class FoldSplit:
    """One fold's disjoint training, selection-validation and optional holdout animals."""

    index: int
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str] = field(default_factory=list)

    def validate(self, all_ids: set[str]) -> None:
        """Ensure no duplicated, unknown, or overlapping subjects occur."""
        if not self.train_ids or not self.val_ids:
            raise ValueError(f"Fold {self.index}: training and validation cannot be empty")
        selected = self.train_ids + self.val_ids + self.test_ids
        roles = (self.train_ids, self.val_ids, self.test_ids)
        if any(len(role) != len(set(role)) for role in roles):
            raise ValueError(f"Fold {self.index}: duplicate IDs")
        if len(selected) != len(set(selected)):
            raise ValueError(f"Fold {self.index}: subject leakage")
        if set(selected) != all_ids:
            raise ValueError(f"Fold {self.index}: split roles must cover all subjects")

    def to_dict(self) -> dict[str, list[str]]:
        """Serialize the roles without implying validation is an independent test."""
        payload = {"train": self.train_ids, "validation": self.val_ids}
        if self.test_ids:
            payload["test"] = self.test_ids
        return payload

    def save(self, folder: Path) -> None:
        """Save the exact subject split for reproducibility."""
        payload = self.to_dict()
        (folder / "split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class FoldContext:
    """Context passed to a custom Keras callback factory.

    Attributes:
        split: Subject identifiers and index for the current fold.
        output_dir: Output directory belonging only to this fold.
        config: Experiment settings.
        checkpoint_path: Runner-managed best-weights checkpoint path.
    """

    split: FoldSplit
    output_dir: Path
    config: TrainingConfig
    checkpoint_path: Path


@dataclass
class TrainingResult:
    """Run folder, training histories, and optional full-volume results."""

    run_dir: Path
    patch_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    classifier_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    histories: list[History] = field(default_factory=list)
    evaluations: dict[str, list[pd.DataFrame]] = field(default_factory=dict)
    summaries: dict[str, float] = field(default_factory=dict)
    timings: list[FoldTiming] = field(default_factory=list)
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)


def _named_evaluators(items: Evaluators | None) -> dict[str, EvaluationFactory]:
    """Convert optional evaluation callbacks to a named dictionary."""
    if items is None:
        return {}
    named = dict(items) if isinstance(items, Mapping) else {
        f"evaluation_{i}": factory for i, factory in enumerate(items)
    }
    for name, factory in named.items():
        _safe_name(name)
        if name == "summary" or re.fullmatch(r"fold_\d+", name):
            raise ValueError(f"Evaluator name collides with a runner directory: {name!r}")
        if not callable(factory):
            raise TypeError(f"Evaluator {name!r} must be callable")
    return named


def _splits(ids: list[str], config: TrainingConfig, splitter: SplitFactory | None) -> list[FoldSplit]:
    """Build and validate every fold before training any model."""
    if splitter is None:
        fold_splitter = KFold(config.k, shuffle=True, random_state=config.seed)
        source = fold_splitter.split(ids)
    else:
        source = splitter(ids)

    pairs = list(source)
    if len(pairs) != config.k:
        raise ValueError(f"Expected {config.k} folds, got {len(pairs)}")

    folds: list[FoldSplit] = []
    validation_counts = {subject: 0 for subject in ids}
    for index, (train_indices, val_indices) in enumerate(pairs):
        train_ids = _subjects_at_indices(ids, train_indices)
        val_ids = _subjects_at_indices(ids, val_indices)
        split = FoldSplit(index, train_ids, val_ids)
        split.validate(set(ids))
        folds.append(split)
        for subject in val_ids:
            validation_counts[subject] += 1

    if any(count != 1 for count in validation_counts.values()):
        raise ValueError("Every subject must appear in validation exactly once across folds")
    return folds


def _subjects_at_indices(ids: list[str], indices: Sequence[int]) -> list[str]:
    """Reject negative, fractional or boolean indices from custom splitters."""
    selected: list[str] = []
    for index in indices:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise ValueError("Split indices must be integers")
        if not 0 <= index < len(ids):
            raise ValueError(f"Split index out of range: {index}")
        selected.append(ids[index])

    return selected


def load_fold_splits(path: str | Path, ids: list[str], k: int) -> list[FoldSplit]:
    """Read precomputed splits in JSON format and validate full CV coverage.

    Expected format: [{"train": ["s1"], "validation": ["s2"], "test": ["s3"]}, ...].
    Optional test lists represent independent outer holdouts. Without them,
    validation lists provide diagnostic coverage and checkpoint selection.
    Keys ``train_ids`` and ``val_ids`` are accepted for interoperability.
    """
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) != k:
        raise ValueError(f"Split file must contain exactly {k} folds")
    folds = []
    counts = dict.fromkeys(ids, 0)
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Fold {index}: expected a JSON object")
        train_ids = entry.get("train", entry.get("train_ids"))
        val_ids = entry.get("validation", entry.get("val_ids"))
        test_ids = entry.get("test", [])
        if not isinstance(train_ids, list) or not isinstance(val_ids, list):
            raise ValueError(f"Fold {index}: missing train/validation lists")
        if not isinstance(test_ids, list) or any(not isinstance(subject, str) for subject in train_ids + val_ids + test_ids):
            raise ValueError(f"Fold {index}: subject IDs must be strings")
        split = FoldSplit(index, train_ids, val_ids, test_ids)
        split.validate(set(ids))
        folds.append(split)
        evaluation_ids = val_ids
        if test_ids:
            evaluation_ids = test_ids
        for subject in evaluation_ids:
            counts[subject] += 1
    if any(count != 1 for count in counts.values()):
        raise ValueError("Each CV subject must appear in the evaluated fold role once")
    if any(fold.test_ids for fold in folds) and not all(fold.test_ids for fold in folds):
        raise ValueError("Cannot mix selection-only folds and independent holdout folds")
    return folds


def _runtime_metadata() -> dict[str, Any]:
    """Collect runtime version and accelerator details without requiring TensorFlow."""
    details: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import tensorflow as tf
        details["tensorflow"] = tf.__version__
        details["devices"] = [device.name for device in tf.config.list_physical_devices()]
    except ImportError:
        details["tensorflow"] = None
        details["devices"] = []
    return details


def _record_evaluation(
    result: TrainingResult, name: str, role: str, fold: int,
    table: pd.DataFrame, requested_ids: Sequence[str],
) -> None:
    """Store evaluator data with explicit model fold and cohort labels."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Evaluator {name!r} must return a pandas DataFrame")
    if "Subject" not in table.columns and table.index.name == "Subject":
        table = table.reset_index()
    if {"Metric", "Value"}.issubset(table.columns) and "Subject" not in table.columns:
        raise ValueError(f"Evaluator {name!r} must identify subjects in long-form metric rows")
    managed_columns = {"Fold", "DatasetRole", "Evaluator", "ModelVariant", "ExperimentName", "DatasetName", "RunID"}
    reserved = managed_columns & set(table.columns)
    if reserved:
        raise ValueError(f"Evaluator {name!r} returned reserved columns: {reserved}")
    if "Subject" in table.columns:
        if table["Subject"].isna().any():
            raise ValueError(f"Evaluator {name!r} returned missing subject IDs")
        observed = set(table["Subject"].dropna().astype(str))
        if not observed.issubset(set(requested_ids)):
            raise ValueError(f"Evaluator {name!r} returned unexpected subject IDs")
        if {"Metric", "Value"}.issubset(table.columns) and observed != set(requested_ids):
            raise ValueError(f"Evaluator {name!r} omitted requested subjects")

    enriched = table.assign(
        Fold=fold, DatasetRole=role, Evaluator=name,
        ModelVariant=result.metadata.model_variant,
        ExperimentName=result.metadata.experiment_name,
        DatasetName=result.metadata.dataset_name,
        RunID=result.run_dir.name,
    )
    result.evaluations.setdefault(name, []).append(enriched)


def _best_epoch(history: History, monitor: str, mode: str) -> int:
    """Select the best epoch using the same objective as ModelCheckpoint."""
    values = np.asarray(history.get(monitor, []), dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"Metric {monitor!r} is missing, empty, or nonfinite")
    return int(np.argmin(values) if mode == "min" else np.argmax(values))


def _build_callbacks(
    context: FoldContext, factory: CallbackFactory | None, *, resume: bool = False,
) -> list[Any]:
    """Create fresh callbacks per fold, always appending the runner checkpoint."""
    import tensorflow as tf

    config = context.config
    folder = context.output_dir
    callbacks: list[Any] = []
    if config.default_callbacks:
        callbacks.extend([
            tf.keras.callbacks.TensorBoard(log_dir=str(folder / "tensorboard"), histogram_freq=config.histogram_freq),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=config.monitor, mode=config.monitor_mode, factor=0.5, patience=10),
            tf.keras.callbacks.CSVLogger(str(folder / "training.csv"), append=resume),
        ])
    if factory is not None:
        callbacks.extend(factory(context))

    for callback in callbacks:
        if isinstance(callback, tf.keras.callbacks.EarlyStopping) and callback.restore_best_weights:
            raise ValueError("The runner restores its selected checkpoint; use restore_best_weights=False")
        if isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
            requested_path = Path(callback.filepath).resolve()
            managed_path = context.checkpoint_path.resolve()
            if requested_path == managed_path:
                raise ValueError("An extra ModelCheckpoint must use a different file from the managed best checkpoint")

    # The checkpoint is appended last, and always uses the same monitor as reporting.
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=str(context.checkpoint_path), monitor=config.monitor,
        mode=config.monitor_mode, save_best_only=True, save_weights_only=True,
    ))
    return callbacks


def _plot_curves(histories: Sequence[History], keys: Sequence[str], folder: Path, name: str) -> None:
    """Write a Plotly HTML figure for selected history keys."""
    figure = go.Figure()
    for fold, history in enumerate(histories):
        for key in keys:
            if key in history:
                label = f"Fold {fold}: {key}" if len(histories) > 1 else key
                figure.add_trace(go.Scatter(y=history[key], name=label, mode="lines"))
    if figure.data:
        figure.update_layout(title=name.replace("_", " "), xaxis_title="Epoch (0-based)")
        figure.write_html(folder / f"{_safe_name(name)}.html", include_plotlyjs="cdn")


def _save_history(history: History, folder: Path) -> None:
    """Export fold history and its training/validation curves."""
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    _plot_curves([history], ["loss", "val_loss"], folder, "loss")
    metrics = [key for key in history if not key.startswith("val_") and key != "loss"]
    keys = [key for metric in metrics for key in (metric, f"val_{metric}")]
    _plot_curves([history], keys, folder, "metrics")


def _save_summary(result: TrainingResult, config: TrainingConfig, metrics: Sequence[str]) -> None:
    """Aggregate training histories independently of optional evaluators."""
    folder = result.run_dir / "summary"
    folder.mkdir(exist_ok=True)
    selected: list[dict[str, float | None]] = []
    keys = [config.monitor, "val_loss", *(f"val_{key}" for key in metrics)]

    for history in result.histories:
        epoch = _best_epoch(history, config.monitor, config.monitor_mode)
        selected.append({key: history[key][epoch] for key in dict.fromkeys(keys) if key in history})

    best = pd.DataFrame(selected)
    best.index.name = "Fold"
    best.to_csv(folder / "selected_epoch_metrics.csv")
    result.summaries = {key: float(value) for key, value in best.mean(numeric_only=True).items()}
    (folder / "average_metrics.json").write_text(json.dumps(result.summaries, indent=2), encoding="utf-8")

    _plot_curves(result.histories, ["loss", "val_loss"], folder, "loss_by_fold")
    _plot_curves(result.histories, [f"val_{key}" for key in metrics], folder, "validation_metrics_by_fold")


def _save_evaluations(result: TrainingResult) -> None:
    """Export per-subject evaluation rows suitable for paired comparisons."""
    all_tables = []
    for name, tables in result.evaluations.items():
        folder = result.run_dir / _safe_name(name)
        folder.mkdir(exist_ok=True)
        combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        combined.to_csv(folder / "full_volume_metrics.csv", index=False)
        if not combined.empty:
            combined.to_excel(folder / "full_volume_metrics.xlsx", index=False)
            all_tables.append(combined)
    if all_tables:
        pd.concat(all_tables, ignore_index=True).to_csv(
            result.run_dir / "summary" / "all_evaluations.csv", index=False
        )
    from ..experiments.evaluation import export_evaluation_tables

    full_volume = pd.DataFrame()
    if all_tables:
        full_volume = pd.concat(all_tables, ignore_index=True)
    from .reporting import split_scan_manifest

    manifest = pd.DataFrame()
    split_manifest = pd.DataFrame()
    manifest_path = result.run_dir / "subjects.csv"
    split_path = result.run_dir / "splits.json"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if split_path.exists():
            saved_splits = json.loads(split_path.read_text(encoding="utf-8"))
            split_manifest = split_scan_manifest(manifest, saved_splits)
    export_evaluation_tables(
        result.patch_metrics, full_volume, result.run_dir / "summary",
        classifier_predictions=result.classifier_predictions, manifest=manifest, split_manifest=split_manifest,
    )



def train_fold(
    context: FoldContext,
    train_factory: DatasetFactory,
    val_factory: DatasetFactory,
    model_factory: ModelFactory,
    callback_factory: CallbackFactory | None,
    evaluators: Mapping[str, EvaluationFactory],
    evaluation_config: Mapping[str, Any],
    result: TrainingResult,
    evaluation: EvaluationConfig,
    model_exporter: Callable[[Any, Path], None] | None = None,
    resume: bool = False,
    patch_evaluator: Callable[[Any, Sequence[str], Mapping[str, Any]], pd.DataFrame] | None = None,
) -> None:
    """Train a fresh model, restore its best checkpoint and evaluate one fold."""
    import tensorflow as tf
    from .training_state import EpochCheckpoint, write_json

    split = context.split
    config = context.config
    folder = context.output_dir
    if resume and (folder / "split.json").exists():
        saved = json.loads((folder / "split.json").read_text(encoding="utf-8"))
        if saved != split.to_dict():
            raise ValueError("Resume split differs from the saved fold")
    else:
        split.save(folder)

    completion_path = folder / "completed.json"
    if resume and completion_path.exists():
        required = [context.checkpoint_path, folder / "history.csv", folder / "timing.json"]
        if config.save_model:
            required.append(folder / "model.keras")
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Completed fold is missing an artifact: {path}")
        completed = json.loads(completion_path.read_text(encoding="utf-8"))
        patch_path = completed.get("patch_metrics")
        if patch_evaluator is not None and patch_path is None:
            raise ValueError("Completed fold predates patch evaluation; evaluate its best checkpoint separately")
        if patch_path is not None:
            patch_table = pd.read_json(folder / patch_path, orient="table")
            result.patch_metrics = pd.concat([result.patch_metrics, patch_table], ignore_index=True)
        result.histories.append(completed["history"])
        result.timings.append(FoldTiming(**completed["timing"]))
        for entry in completed["evaluations"]:
            table = pd.read_json(folder / entry["path"], orient="table")
            _record_evaluation(result, entry["name"], entry["role"], split.index, table, entry["subjects"])
            classifier_path = entry.get("classifier_predictions")
            if classifier_path is not None:
                classifier_table = pd.read_json(folder / classifier_path, orient="table")
                result.classifier_predictions = pd.concat(
                    [result.classifier_predictions, classifier_table], ignore_index=True,
                )
        return
    if resume and context.checkpoint_path.exists() and not (folder / "training_state.json").exists():
        raise ValueError("Best weights alone cannot resume training; epoch recovery state is missing")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed + split.index)

    train_data = train_factory(split.train_ids)
    val_data = val_factory(split.val_ids)
    for role, dataset in (("training", train_data), ("validation", val_data)):
        if isinstance(dataset, tf.keras.utils.Sequence) and len(dataset) == 0:
            raise ValueError(f"Fold {split.index}: {role} generator has no complete batches; reduce batch_size")

    model = model_factory()
    initialization = getattr(model, "initialization_report", None)
    if initialization is not None:
        write_json(folder / "initialization.json", initialization)

    callbacks = _build_callbacks(context, callback_factory, resume=resume)
    state = EpochCheckpoint(folder, callbacks, resume=resume, monitor=config.monitor)
    state.set_model(model)
    callbacks.append(state)
    steps = config.fit_steps(len(split.train_ids), len(split.val_ids))
    fit_options = {**config.fit_kwargs, **steps}

    LOGGER.info("Fold %d: %d training, %d validation subjects", split.index, len(split.train_ids), len(split.val_ids))
    if isinstance(val_data, tf.keras.utils.Sequence) and config.validation_steps is not None:
        if config.validation_steps != len(val_data):
            raise ValueError("validation_steps must cover every validation batch exactly once")
    if not state.fit_complete and state.completed_epochs < config.epochs:
        model.fit(
            train_data, validation_data=val_data, epochs=config.epochs,
            initial_epoch=state.completed_epochs, callbacks=callbacks, **fit_options,
        )
    training_seconds = state.training_seconds
    history = state.history
    best_epoch = _best_epoch(history, config.monitor, config.monitor_mode)
    selected_epoch = best_epoch + 1
    if not context.checkpoint_path.is_file():
        raise FileNotFoundError(f"No best-weights checkpoint at {context.checkpoint_path}")

    selected_checkpoint = context.checkpoint_path
    state_file = folder / "training_state.json"
    if state_file.is_file():
        saved_state = json.loads(state_file.read_text(encoding="utf-8"))
        selected_checkpoint = folder / saved_state["selected_checkpoint"]
    model.load_weights(str(selected_checkpoint))
    if config.save_model:
        model_path: Path = folder / "model.keras"
        if model_exporter is None:
            model.save(str(model_path))
        else:
            model_exporter(model, model_path)
    _save_history(history, folder)
    result.histories.append(history)

    eval_start = time.perf_counter()
    saved_evaluations: list[dict[str, Any]] = []
    patch_path = None
    if patch_evaluator is not None:
        patch_tables = []
        patch_cohorts = {"validation": split.val_ids, "holdout": split.test_ids,
                         "test": evaluation.test_ids, "external": evaluation.external_ids}
        for role, subject_ids in patch_cohorts.items():
            if not subject_ids:
                continue
            patch_dir = folder / "evaluations" / role / "patches"
            patch_options = {"evaluation_dir": patch_dir}
            patch_table = patch_evaluator(model, subject_ids, patch_options)
            patch_table = patch_table.assign(
                Fold=split.index, DatasetRole=role, Seed=config.seed, BestEpoch=selected_epoch,
                ModelVariant=result.metadata.model_variant, ExperimentName=result.metadata.experiment_name,
                RunID=result.run_dir.name,
            )
            patch_tables.append(patch_table)
        if patch_tables:
            patch_table = pd.concat(patch_tables, ignore_index=True)
            patch_path = "patch_metrics.json"
            patch_table.to_json(folder / patch_path, orient="table", indent=2)
            result.patch_metrics = pd.concat([result.patch_metrics, patch_table], ignore_index=True)
    cohorts = {
        "validation": split.val_ids if evaluation.evaluate_validation else (),
        "holdout": split.test_ids,
        "test": evaluation.test_ids,
        "external": evaluation.external_ids,
    }
    for role, subject_ids in cohorts.items():
        if not subject_ids:
            continue
        for name, evaluator_factory in evaluators.items():
            evaluator_dir: Path = folder / "evaluations" / role / name
            evaluator_dir.mkdir(parents=True, exist_ok=True)
            options = {**evaluation_config, "dataset_role": role,
                       "save_predictions": evaluation.save_predictions,
                       "fold": split.index, "evaluation_dir": str(evaluator_dir)}
            selected_ids: list[str] = list(subject_ids)
            evaluator = evaluator_factory(model, selected_ids, options)
            table = evaluator.evaluate()
            table = table.assign(Seed=config.seed, BestEpoch=selected_epoch)
            _record_evaluation(result, name, role, split.index, table, subject_ids)
            table_path = evaluator_dir / "metrics.json"
            table.to_json(table_path, orient="table", indent=2)
            saved_evaluations.append({
                "name": name, "role": role, "subjects": selected_ids,
                "path": str(table_path.relative_to(folder)),
            })
            classifier_table = getattr(evaluator, "classifier_predictions", None)
            if classifier_table is not None and not classifier_table.empty:
                classifier_table = classifier_table.assign(
                    Fold=split.index, DatasetRole=role, Seed=config.seed, BestEpoch=selected_epoch,
                    Variant=result.metadata.model_variant, RunID=result.run_dir.name,
                )
                result.classifier_predictions = pd.concat(
                    [result.classifier_predictions, classifier_table], ignore_index=True,
                )
                classifier_path = evaluator_dir / "classifier_predictions.json"
                classifier_table.to_json(classifier_path, orient="table", indent=2)
                classifier_table.to_csv(evaluator_dir / "classifier_predictions.csv", index=False)
                saved_evaluations[-1]["classifier_predictions"] = str(classifier_path.relative_to(folder))

    timing = FoldTiming(
        fold=split.index,
        training_seconds=training_seconds,
        evaluation_seconds=time.perf_counter() - eval_start,
        epochs_completed=len(history[config.monitor]),
        parameter_count=int(model.count_params()),
        best_epoch=best_epoch,
    )
    result.timings.append(timing)
    (folder / "timing.json").write_text(json.dumps(asdict(timing), indent=2), encoding="utf-8")
    write_json(completion_path, {"history": history, "timing": asdict(timing),
                                "evaluations": saved_evaluations, "patch_metrics": patch_path})

    del model, train_data, val_data
    gc.collect()


def save_training_result(result: TrainingResult, config: TrainingConfig) -> None:
    """Export common histories, evaluation tables and timings for any run type."""
    _save_summary(result, config, metrics=())
    _save_evaluations(result)
    timings = pd.DataFrame(asdict(timing) for timing in result.timings)
    timings.to_csv(result.run_dir / "summary" / "fold_timings.csv", index=False)


def k_fold_loop(
    train_gen: DatasetFactory,
    val_gen: DatasetFactory,
    model_callback: ModelFactory,
    config: TrainingConfig | Mapping[str, Any],
    train_and_test_ids: Sequence[str],
    fve_callbacks: Evaluators | None = None,
    metrics_dictionary: Mapping[str, Any] | None = None,
    *,
    callbacks_factory: CallbackFactory | None = None,
    splitter: SplitFactory | None = None,
    precomputed_splits: str | Path | None = None,
    evaluation: EvaluationConfig | None = None,
    metadata: ExperimentMetadata | None = None,
    subject_index: SubjectIndex | None = None,
    model_exporter: Callable[[Any, Path], None] | None = None,
    split_seed: int | None = None,
    run_setup: Callable[[Path], None] | None = None,
    resume_run: str | Path | None = None,
    fold_splits: Sequence[FoldSplit] | None = None,
    fold_validator: Callable[[FoldSplit], None] | None = None,
    fold_model_factory: Callable[[FoldContext], Any] | None = None,
    patch_evaluator: Callable[[Any, Sequence[str], Mapping[str, Any]], pd.DataFrame] | None = None,
) -> TrainingResult:
    """Run a fresh TensorFlow/Keras training and validation experiment per fold.

    Args:
        train_gen: Factory mapping fold training IDs to a fresh dataset.
        val_gen: Factory mapping fold validation IDs to a fresh dataset.
        model_callback: Factory returning a freshly initialized *compiled* model.
        config: Dataclass or mapping describing run settings.
        train_and_test_ids: Unique CV subject identifiers, excluding independent
            test/external animals. The historical argument name is retained.
        fve_callbacks: Optional named full-volume evaluator factories. Each
            receives ``(model, validation_ids, config_mapping)`` and returns
            an object providing ``evaluate() -> pandas.DataFrame``.
        metrics_dictionary: Optional metric names (the mapping values are unused).
        callbacks_factory: Optional factory receiving a :class:`FoldContext`
            and returning fresh Keras callbacks for that fold. Standard callbacks
            are controlled by ``config.default_callbacks``; the managed best-
            weights checkpoint is always appended.
        precomputed_splits: JSON file containing reusable subject-level folds.
            Mutually exclusive with ``splitter``.
        evaluation: Optional independent test/external cohorts and evaluation options.
        metadata: Experiment provenance including dataset and model variant.
        subject_index: Optional loader identity index. Validates all cohorts and
            saves their subject-to-scan manifest. Factories still receive subject
            IDs; expand them with ``subject_index.scan_keys(ids)`` for generators.
        splitter: Optional factory receiving IDs and yielding pairs of integer
            training/validation indices. Must produce exactly ``config.k``
            disjoint folds, with each subject validated exactly once.
        model_exporter: Optional inference-graph exporter after best-weight restoration.
        split_seed: Optional split RNG seed independent of model initialization.
        run_setup: Optional provenance writer invoked before fitting any fold.
        resume_run: Existing matching run; restore saved splits and skip completed folds.
        fold_splits: Explicit grouped roles, optionally including independent
            outer holdouts. Mutually exclusive with split files and a splitter.
        fold_validator: Preflight every fold before creating a run or fitting.
        fold_model_factory: Optional context-aware model factory for fold-specific initialization.
        patch_evaluator: Native compiled-model measurements on complete deterministic patch cohorts.

    Returns:
        TrainingResult: Run location, per-fold histories and evaluations, and
        mean selected-epoch metrics.
    """
    if not all(callable(factory) for factory in (train_gen, val_gen, model_callback)):
        raise TypeError("Dataset and model factories must be callable")

    ids: list[str] = list(train_and_test_ids)
    if any(not isinstance(subject, str) or not subject.strip() for subject in ids):
        raise ValueError("CV IDs must be nonempty subject strings; use SubjectIndex for loader scan keys")
    if len(ids) != len(set(ids)):
        raise ValueError("train_and_test_ids must contain unique subjects")

    settings = TrainingConfig.from_input(config)
    settings.validate(len(ids))
    choices = (splitter is not None, precomputed_splits is not None, fold_splits is not None)
    if sum(choices) > 1:
        raise ValueError("Provide either splitter or precomputed_splits, not both")
    if evaluation is None:
        evaluation = EvaluationConfig()
    evaluation.validate(ids)
    if subject_index is not None:
        selected_subjects: list[str] = ids + list(evaluation.test_ids) + list(evaluation.external_ids)
        subject_index.scan_keys(selected_subjects)
        if settings.infer_steps_from_ids:
            raise ValueError("SubjectIndex expands animals to scans; use generator cardinality or explicit steps")

    resume: bool = resume_run is not None
    if resume:
        run_dir = Path(resume_run).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        if splitter is not None:
            raise ValueError("Resume uses the saved splits, not a new splitter")
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        current_config = {**asdict(settings), "run_id": run_dir.name}
        current_config = json.loads(json.dumps(current_config, default=str))
        if saved_config != current_config:
            raise ValueError("Resume training configuration differs from the saved run")
        saved_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        current_evaluation = json.loads(json.dumps(asdict(evaluation)))
        if saved_metadata["evaluation"] != current_evaluation:
            raise ValueError("Resume evaluation cohorts or options differ")
        folds = load_fold_splits(run_dir / "splits.json", ids, settings.k)
        if precomputed_splits is not None:
            requested = load_fold_splits(precomputed_splits, ids, settings.k)
            if requested != folds:
                raise ValueError("Resume splits differ from the saved run")
        if fold_splits is not None and list(fold_splits) != folds:
            raise ValueError("Resume folds differ from the saved run")
    elif fold_splits is not None:
        folds = list(fold_splits)
        if len(folds) != settings.k or [fold.index for fold in folds] != list(range(settings.k)):
            raise ValueError("Fold indexes must cover the configured folds in order")
        counts = dict.fromkeys(ids, 0)
        has_holdout = any(fold.test_ids for fold in folds)
        for fold in folds:
            fold.validate(set(ids))
            evaluation_ids = fold.val_ids
            if has_holdout:
                if not fold.test_ids:
                    raise ValueError("Every outer fold needs independent holdout animals")
                evaluation_ids = fold.test_ids
            for subject in evaluation_ids:
                counts[subject] += 1
        if any(count != 1 for count in counts.values()):
            raise ValueError("Every animal must appear in the evaluated fold role exactly once")
    elif precomputed_splits is None:
        split_settings = settings
        if split_seed is not None:
            split_settings = replace(settings, seed=split_seed)
        folds = _splits(ids, split_settings, splitter)
    else:
        folds = load_fold_splits(precomputed_splits, ids, settings.k)
    evaluators = _named_evaluators(fve_callbacks)
    metrics = list(metrics_dictionary or {})
    if fold_validator is not None:
        for fold in folds:
            fold_validator(fold)

    if not resume:
        run_dir = settings.start_run()
    from .training_state import run_lock

    with run_lock(run_dir):
        if subject_index is not None:
            manifest: pd.DataFrame = subject_index.manifest(selected_subjects)
            roles = dict.fromkeys(ids, "cv")
            roles.update(dict.fromkeys(evaluation.test_ids, "test"))
            roles.update(dict.fromkeys(evaluation.external_ids, "external"))
            manifest["DatasetRole"] = manifest["Subject"].map(roles)
            manifest_path: Path = run_dir / "subjects.csv"
            if resume:
                saved_manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
                current_manifest = manifest.fillna("").astype(str)
                if "Subgroup" not in saved_manifest.columns:
                    current_manifest = current_manifest.drop(columns="Subgroup")
                if not saved_manifest.equals(current_manifest):
                    raise ValueError("Resume subject-to-scan manifest differs")
            else:
                manifest.to_csv(manifest_path, index=False)

        evaluation_config = {**asdict(settings), "run_id": run_dir.name}
        if not resume:
            (run_dir / "config.json").write_text(json.dumps(evaluation_config, indent=2, default=str), encoding="utf-8")
        provenance = {
            "experiment": asdict(metadata or ExperimentMetadata()),
            "runtime": _runtime_metadata(),
            "evaluation": asdict(evaluation),
            "cv_subject_count": len(ids),
            "cv_subject_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
            "precomputed_splits": str(precomputed_splits) if precomputed_splits else None,
            "split_seed": settings.seed if split_seed is None else split_seed,
        }
        if not resume:
            (run_dir / "metadata.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        split_payload = [fold.to_dict() for fold in folds]
        if not resume:
            (run_dir / "splits.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
        if run_setup is not None:
            run_setup(run_dir)
        experiment_metadata = metadata or ExperimentMetadata()
        result = TrainingResult(
            run_dir=run_dir, evaluations={name: [] for name in evaluators},
            metadata=experiment_metadata,
        )

        for split in folds:
            folder = run_dir / f"fold_{split.index}"
            folder.mkdir(exist_ok=resume)
            checkpoint_path = folder / "best.weights.h5"
            context = FoldContext(split=split, output_dir=folder, config=settings, checkpoint_path=checkpoint_path)

            selected_model_factory = model_callback
            if fold_model_factory is not None:
                def selected_model_factory() -> Any:
                    """Build the current fold with its explicit initialization."""
                    return fold_model_factory(context)

            train_fold(
                context=context,
                train_factory=train_gen,
                val_factory=val_gen,
                model_factory=selected_model_factory,
                callback_factory=callbacks_factory,
                evaluators=evaluators,
                evaluation_config=evaluation_config,
                result=result,
                evaluation=evaluation,
                model_exporter=model_exporter,
                resume=resume,
                patch_evaluator=patch_evaluator,
            )

        _save_summary(result, settings, metrics)
        _save_evaluations(result)
        pd.DataFrame(asdict(timing) for timing in result.timings).to_csv(
            run_dir / "summary" / "fold_timings.csv", index=False
        )
        return result
