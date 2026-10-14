"""Audited complete loading or feature transfer across different label vocabularies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

from .config import ExperimentConfig, InitializationConfig

if TYPE_CHECKING:
    from .data import PreparedData


def pretraining_audit(
    data: PreparedData, settings: InitializationConfig, protected_ids: Sequence[str],
    *, ancestors: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Reject target validation/test animals used in source fitting or selection.

    Require saved source splits and a scan manifest, and inspect initialization
    ancestors too. Match canonical animal IDs, exact original IDs within a subgroup,
    and image paths. Explicit animal overrides connect confirmed cross-cohort IDs.
    """
    if settings.mode == "scratch":
        return {"mode": "scratch"}

    checkpoint = Path(settings.checkpoint).resolve()
    if checkpoint in ancestors:
        raise ValueError("Pretraining provenance contains an initialization cycle")
    run_dir: Path | None = None
    if settings.source_experiment is not None:
        run_dir = Path(settings.source_experiment).resolve().parent
    else:
        for parent in checkpoint.parents:
            if (parent / "experiment.json").is_file():
                run_dir = parent
                break
    if run_dir is None or not checkpoint.is_relative_to(run_dir):
        raise ValueError("Pretraining requires a saved source run with splits and subjects.csv")
    if settings.source_experiment is not None:
        if Path(settings.source_experiment).resolve() != run_dir / "experiment.json":
            raise ValueError("Pretraining must use the source run's saved experiment.json")

    fold_dir: Path | None = None
    for parent in checkpoint.parents:
        if parent == run_dir:
            break
        if parent.name.startswith("fold_") and parent.name[5:].isdigit():
            fold_dir = parent
            break
    split_file = run_dir / "splits.json"
    if fold_dir is not None:
        split_file = fold_dir / "split.json"
    manifest_file = run_dir / "subjects.csv"
    if not split_file.is_file() or not manifest_file.is_file():
        raise ValueError("Pretraining requires source split.json and subjects.csv for the selected fold")
    split = json.loads(split_file.read_text(encoding="utf-8"))
    if not isinstance(split, dict) or not split.get("train") or not split.get("validation"):
        raise ValueError("Pretraining source must declare training and checkpoint-selection animals")
    used_ids = set(split["train"]) | set(split["validation"])
    source_manifest = pd.read_csv(manifest_file, dtype=str).fillna("")
    required = {"Subject", "Dataset", "OriginalID", "ImagePath"}
    if not required.issubset(source_manifest.columns):
        raise ValueError("Pretraining source manifest is missing animal identity columns")
    if "Subgroup" not in source_manifest.columns:
        # Older manifests omitted subgroup, but their immutable data contract has it.
        subgroups: dict[tuple[str, str], str] = {}
        contract_file = run_dir / "data_contract.json"
        if contract_file.is_file():
            records = json.loads(contract_file.read_text(encoding="utf-8"))
            for record in records:
                key = tuple(record["scan"])
                subgroups[key] = record["metadata"].get("subgroup", "")
        source_manifest["Subgroup"] = ""
        if {"Group", "Scan"}.issubset(source_manifest.columns):
            for index, row in source_manifest.iterrows():
                key = (row["Group"], row["Scan"])
                source_manifest.loc[index, "Subgroup"] = subgroups.get(key, "")
    source = source_manifest.loc[source_manifest["Subject"].isin(used_ids)]
    if set(source["Subject"]) != used_ids:
        raise ValueError("Pretraining source manifest does not cover fitting and selection animals")
    target = data.subjects.manifest(list(protected_ids)).fillna("")
    source_ids = set(source["Subject"])
    source_originals = set(zip(source["Dataset"], source["Subgroup"], source["OriginalID"]))
    unqualified = source.loc[source["Subgroup"] == ""]
    unqualified_originals = set(zip(unqualified["Dataset"], unqualified["OriginalID"]))
    source_paths = {str(Path(path).resolve()).casefold() for path in source["ImagePath"]}
    overlaps = []
    for row in target.itertuples():
        original = (row.Dataset, row.Subgroup, row.OriginalID)
        if row.Subgroup and (row.Dataset, row.OriginalID) in unqualified_originals:
            if row.Subject not in source_ids:
                raise ValueError("Pretraining original-ID matching requires saved subgroup provenance")
        image_path = str(Path(row.ImagePath).resolve()).casefold()
        if row.Subject in source_ids or original in source_originals or image_path in source_paths:
            overlaps.append(row.Subject)
    if overlaps:
        animals = ", ".join(sorted(set(overlaps)))
        raise ValueError(f"Pretraining leakage: target validation/test animals were used by the source: {animals}")

    experiment_file = run_dir / "experiment.json"
    source_config = ExperimentConfig.load(experiment_file)
    normalized_target = any(dataset.layout == "normalized" for dataset in data.config.data.datasets)
    legacy_modalities = {dataset.modality for dataset in source_config.data.datasets if dataset.layout == "legacy"}
    if normalized_target and legacy_modalities:
        legacy_source = source.loc[source["Dataset"].isin(legacy_modalities)]
        confirmed_animals = set(source_config.data.animal_ids.values())
        if not set(legacy_source["Subject"]).issubset(confirmed_animals):
            raise ValueError("Legacy pretraining sources need explicit animal_ids links to the normalized inventory")
    if settings.mode == "backbone":
        fields = ("architecture", "filters", "residual", "attention", "max_pooling")
        source_graph = tuple(getattr(source_config.model, name) for name in fields)
        target_graph = tuple(getattr(data.config.model, name) for name in fields)
        if source_graph != target_graph:
            raise ValueError("Source and target backbone settings differ; use a variant-matched TBI source")
    source_fold = 0
    if fold_dir is not None:
        source_fold = int(fold_dir.name[5:])
    parent_settings = source_config.initialization_by_fold.get(source_fold, source_config.initialization)
    # Audit ancestors against their immediate target architecture, too.
    from dataclasses import replace

    parent_config = replace(data.config, model=source_config.model)
    parent_data = replace(data, config=parent_config)
    next_ancestors = (*ancestors, checkpoint)
    parent_audit = pretraining_audit(parent_data, parent_settings, protected_ids, ancestors=next_ancestors)
    files = [checkpoint, split_file, manifest_file, experiment_file]
    contract_file = run_dir / "data_contract.json"
    if contract_file.is_file():
        files.append(contract_file)
    digests: dict[str, str] = {}
    for path in files:
        content = path.read_bytes()
        digests[str(path)] = hashlib.sha256(content).hexdigest()
    return {
        "mode": settings.mode, "source_run": str(run_dir), "source_fold": source_fold,
        "used_animals": sorted(used_ids), "protected_animals": sorted(protected_ids),
        "sha256": digests, "ancestor": parent_audit,
    }


def backbone(model: Any) -> Any:
    """Extract feature ancestors at the existing segmentation-head boundaries."""
    import tensorflow as tf

    names = {layer.name for layer in model.layers}
    if "r_output_1" in names:
        features = [model.get_layer(f"r_output_{level}").input for level in (4, 3, 2, 1)]
    else:
        head_name: str = "regions_logits"
        if "regions" in names:
            head_name = "regions"
        head = model.get_layer(head_name)
        projection = head.input._keras_history.layer
        while isinstance(projection, tf.keras.layers.Activation):
            projection = projection.input._keras_history.layer
        if not isinstance(projection, tf.keras.layers.Conv3D):
            raise ValueError("Unsupported model: cannot identify the ROI projection")
        features = [projection.input]

    return tf.keras.Model(model.inputs, features)


def graph_signature(model: Any, *, for_transfer: bool = False) -> str:
    """Describe topology independently of automatic layer names/input sizes.

    Transfer normalizes graph edges only, preserving literal operation names.
    The default retains existing fitting fingerprints for saved recovery records.
    """
    configuration = model.get_config()
    names: dict[str, str] = {}
    for index, layer in enumerate(configuration["layers"]):
        name = layer["config"]["name"]
        names[name] = f"layer_{index}"

    def references(value: Any) -> Any:
        """Rename graph edges without replacing operation names such as 'lambda'."""
        if isinstance(value, dict):
            return {key: references(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [references(item) for item in value]
        if isinstance(value, str):
            return names.get(value, value)
        return value

    def normalize(value: Any) -> Any:
        """Normalize graph references, retaining operations and connections."""
        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if key in ("name", "trainable", "batch_input_shape"):
                    continue
                if for_transfer and key in ("inbound_nodes", "input_layers", "output_layers", "keras_history"):
                    normalized[key] = references(item)
                else:
                    normalized[key] = normalize(item)
            return normalized
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, str) and not for_transfer:
            return names.get(value, value)
        return value

    signature = normalize(configuration)
    return json.dumps(signature, sort_keys=True, default=str)


def _source_model(settings: InitializationConfig) -> tuple[Any, dict[str, Any]]:
    """Load an export or reconstruct a weights-only graph from saved settings."""
    import tensorflow as tf

    checkpoint = Path(settings.checkpoint)
    report: dict[str, Any] = {}
    if settings.source_experiment is None:
        custom_objects = {"tf": tf, "K": tf.keras.backend, "training": False}
        source = tf.keras.models.load_model(str(checkpoint), compile=False, custom_objects=custom_objects)
    else:
        from .training import build_network, training_view

        source_file = Path(settings.source_experiment)
        config = ExperimentConfig.load(source_file)
        source = build_network(config)
        source = training_view(source, config)
        source.load_weights(str(checkpoint))
        report = {
            "source_experiment": str(source_file.resolve()),
            "source_experiment_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "source_model": config.to_dict()["model"],
            "source_labels": config.to_dict()["labels"],
            "source_reference": str(config.preprocessing.reference),
        }

    return source, report


def initialize_model(model: Any, settings: InitializationConfig) -> dict[str, Any]:
    """Transfer only a fully validated graph; report copied and untouched layers.

    Backbone mode accepts an export or fitting weights with source_experiment.
    ROI, brain and classifier heads remain freshly initialized. Strict weights mode
    uses Keras's complete topology/shape checks and assumes the user verified
    output label semantics. Frozen backbone layers include batch normalization.
    """
    import tensorflow as tf

    settings.validate()
    report: dict[str, Any] = {"mode": settings.mode, "copied": [], "frozen": []}
    if settings.mode == "scratch":
        return report

    path = Path(settings.checkpoint)
    report["checkpoint"] = str(path.resolve())
    report["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    features = backbone(model)
    if settings.mode == "weights":
        model.load_weights(str(path))
        report["copied"] = [layer.name for layer in model.layers if layer.weights]
    else:
        source, source_report = _source_model(settings)
        report.update(source_report)
        source_features = backbone(source)
        source_signature = graph_signature(source_features, for_transfer=True)
        target_signature = graph_signature(features, for_transfer=True)
        if source_signature != target_signature:
            raise ValueError("Source and target backbone graphs differ; transfer was not applied")

        source_layers = [layer for layer in source_features.layers if layer.weights]
        target_layers = [layer for layer in features.layers if layer.weights]
        weights = [layer.get_weights() for layer in source_layers]
        if len(source_layers) != len(target_layers):
            raise ValueError("Backbone weighted-layer counts differ")
        for layer, source_weights in zip(target_layers, weights):
            target_shapes = [weight.shape for weight in layer.get_weights()]
            source_shapes = [weight.shape for weight in source_weights]
            if source_shapes != target_shapes:
                raise ValueError(f"Backbone weight mismatch at {layer.name}; transfer was not applied")

        for source_layer, layer, source_weights in zip(source_layers, target_layers, weights):
            layer.set_weights(source_weights)
            report["copied"].append({"source": source_layer.name, "target": layer.name})
        feature_names = {layer.name for layer in features.layers}
        report["reinitialized_heads"] = [layer.name for layer in model.layers
                                         if layer.weights and layer.name not in feature_names]

    if settings.freeze_backbone:
        for layer in features.layers:
            layer.trainable = False
            if layer.weights:
                report["frozen"].append(layer.name)

    return report
