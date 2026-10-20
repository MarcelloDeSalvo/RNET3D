"""Audited complete loading or feature transfer across different label vocabularies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

from .config import ExperimentConfig, InitializationConfig

if TYPE_CHECKING:
    from .data import PreparedData


@dataclass(frozen=True)
class PretrainingSource:
    """Saved source memberships and scan identities, without loading a model."""

    run_dir: Path
    fold: int
    split_file: Path
    manifest_file: Path
    config: ExperimentConfig
    roles: dict[str, tuple[str, ...]]
    manifest: pd.DataFrame

    def matching_roles(self, data: PreparedData, subjects: Sequence[str]) -> dict[str, set[str]]:
        """Match canonical animals, qualified originals and exact scan paths."""
        source = self.manifest
        source_ids = set(source["Subject"])
        originals: dict[tuple[str, str, str], set[str]] = {}
        paths: dict[str, set[str]] = {}
        unqualified: set[tuple[str, str]] = set()
        for row in source.itertuples():
            original = (row.Dataset, row.Subgroup, row.OriginalID)
            originals.setdefault(original, set()).add(row.Subject)
            path = str(Path(row.ImagePath).resolve()).casefold()
            paths.setdefault(path, set()).add(row.Subject)
            if not row.Subgroup:
                unqualified.add((row.Dataset, row.OriginalID))

        role_by_animal = {animal: role for role, animals in self.roles.items() for animal in animals}
        matches: dict[str, set[str]] = {}
        target = data.subjects.manifest(list(subjects)).fillna("")
        for row in target.itertuples():
            original = (row.Dataset, row.Subgroup, row.OriginalID)
            if row.Subgroup and (row.Dataset, row.OriginalID) in unqualified and row.Subject not in source_ids:
                raise ValueError("Pretraining original-ID matching requires saved subgroup provenance")

            matched = set(originals.get(original, ()))
            path = str(Path(row.ImagePath).resolve()).casefold()
            matched.update(paths.get(path, ()))
            if row.Subject in source_ids:
                matched.add(row.Subject)
            roles = {role_by_animal[animal] for animal in matched if animal in role_by_animal}
            matches.setdefault(row.Subject, set()).update(roles)

        return matches


def load_pretraining_source(settings: InitializationConfig, *, include_test: bool = False) -> PretrainingSource:
    """Read the selected source fold; optionally recover its final test role."""
    settings.validate()
    checkpoint = Path(settings.checkpoint).resolve()
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
    experiment_file = run_dir / "experiment.json"
    if settings.source_experiment is not None and Path(settings.source_experiment).resolve() != experiment_file:
        raise ValueError("Pretraining must use the source run's saved experiment.json")

    source_fold: int = 0
    split_file = run_dir / "splits.json"
    for parent in checkpoint.parents:
        if parent == run_dir:
            break
        if parent.name.startswith("fold_") and parent.name[5:].isdigit():
            source_fold = int(parent.name[5:])
            split_file = parent / "split.json"
            break

    manifest_file = run_dir / "subjects.csv"
    if not split_file.is_file() or not manifest_file.is_file():
        raise ValueError("Pretraining requires source split.json and subjects.csv for the selected fold")
    split = json.loads(split_file.read_text(encoding="utf-8"))
    source_config = ExperimentConfig.load(experiment_file)
    refit_file = run_dir / "refit.json"
    is_refit: bool = refit_file.is_file()
    if not isinstance(split, dict) or not split.get("train"):
        raise ValueError("Pretraining source must declare training and checkpoint-selection animals")
    if not split.get("validation"):
        if is_refit:
            refit = json.loads(refit_file.read_text(encoding="utf-8"))
            valid_refit = (
                refit.get("purpose") == "Deployment refit; no independent evaluation"
                and source_config.training.monitor == "loss"
                and not source_config.training.default_callbacks
                and refit.get("epochs") == source_config.training.epochs
                and not split.get("test")
            )
            if not valid_refit:
                raise ValueError("Refit sources require a fixed-budget, training-only deployment contract")
        elif source_config.training.fixed_epochs:
            state_file = split_file.parent / "training_state.json"
            state: dict[str, Any] = {}
            if state_file.is_file():
                state = json.loads(state_file.read_text(encoding="utf-8"))
            expected_name = f"selected_{source_config.training.epochs}.weights.h5"
            selected_name = Path(state.get("selected_checkpoint", "")).name
            completed = state.get("completed_epochs") == source_config.training.epochs
            valid_fixed = (
                source_config.training.monitor == "loss"
                and not source_config.training.default_callbacks
                and completed and selected_name == expected_name
            )
            if not valid_fixed:
                raise ValueError("Fixed-epoch pretraining requires the completed budget and final selected checkpoint")
        else:
            raise ValueError("Pretraining source must declare training and checkpoint-selection animals")

    root_split = run_dir / "splits.json"
    if include_test and not split.get("test") and root_split != split_file and root_split.is_file():
        complete = json.loads(root_split.read_text(encoding="utf-8"))
        if isinstance(complete, list):
            if source_fold >= len(complete):
                raise ValueError("Selected pretraining fold is missing from splits.json")
            complete = complete[source_fold]
        if not isinstance(complete, dict):
            raise ValueError("Unsupported source split format")
        for role in ("train", "validation"):
            if set(complete.get(role, ())) != set(split[role]):
                raise ValueError("Source fold memberships disagree with splits.json")
        split["test"] = complete.get("test", [])

    roles: dict[str, tuple[str, ...]] = {}
    names = ("train", "validation")
    if include_test:
        names = (*names, "test")
    for role in names:
        animals = split.get(role, [])
        if not isinstance(animals, list) or any(not isinstance(animal, str) or not animal for animal in animals):
            raise ValueError("Source memberships must be lists of animal IDs")
        roles[role] = tuple(animals)
    selected = [animal for animals in roles.values() for animal in animals]
    if len(selected) != len(set(selected)):
        raise ValueError("Pretraining source roles contain duplicated or overlapping animals")

    manifest = pd.read_csv(manifest_file, dtype=str).fillna("")
    required = {"Subject", "Dataset", "OriginalID", "ImagePath"}
    if not required.issubset(manifest.columns):
        raise ValueError("Pretraining source manifest is missing animal identity columns")
    if "Subgroup" not in manifest.columns:
        subgroups: dict[tuple[str, str], str] = {}
        contract_file = run_dir / "data_contract.json"
        if contract_file.is_file():
            records = json.loads(contract_file.read_text(encoding="utf-8"))
            for record in records:
                key = tuple(record["scan"])
                subgroups[key] = record["metadata"].get("subgroup", "")
        manifest["Subgroup"] = ""
        if {"Group", "Scan"}.issubset(manifest.columns):
            for index, row in manifest.iterrows():
                key = (row["Group"], row["Scan"])
                manifest.loc[index, "Subgroup"] = subgroups.get(key, "")
    manifest = manifest.loc[manifest["Subject"].isin(selected)].copy()
    if set(manifest["Subject"]) != set(selected):
        raise ValueError("Pretraining source manifest does not cover declared animals")

    return PretrainingSource(
        run_dir=run_dir, fold=source_fold, split_file=split_file,
        manifest_file=manifest_file, config=source_config, roles=roles, manifest=manifest,
    )


def pretraining_audit(
    data: PreparedData, settings: InitializationConfig, protected_ids: Sequence[str],
    *, validation_ids: Sequence[str] = (), ancestors: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Reject fitted validation animals and any used final-test animals.

    Require saved source splits and a scan manifest, and inspect initialization
    ancestors too. Match canonical animal IDs, exact original IDs within a subgroup,
    and image paths. Explicit animal overrides connect confirmed cross-cohort IDs.
    Explicit validation_ids may reuse source selection-validation, never fitting.
    """
    if settings.mode == "scratch":
        return {"mode": "scratch"}

    checkpoint = Path(settings.checkpoint).resolve()
    if checkpoint in ancestors:
        raise ValueError("Pretraining provenance contains an initialization cycle")
    validation = set(validation_ids)
    if not validation.issubset(protected_ids):
        raise ValueError("Validation IDs must be included in protected animals")

    record = load_pretraining_source(settings)
    run_dir = record.run_dir
    split_file = record.split_file
    manifest_file = record.manifest_file
    experiment_file = run_dir / "experiment.json"
    source_config = record.config
    source = record.manifest
    used_ids = set(record.roles["train"]) | set(record.roles["validation"])
    matches = record.matching_roles(data, protected_ids)
    overlaps: list[str] = []
    shared_validation: list[str] = []
    for animal, roles in matches.items():
        forbidden = {"train", "validation"}
        if animal in validation:
            forbidden = {"train"}
            if "validation" in roles:
                shared_validation.append(animal)
        if forbidden & roles:
            overlaps.append(animal)
    if overlaps:
        animals = ", ".join(sorted(overlaps))
        raise ValueError(f"Pretraining leakage: target validation/test animals were used by the source: {animals}")

    normalized_target = any(dataset.layout == "normalized" for dataset in data.config.data.datasets)
    legacy_modalities = {dataset.modality for dataset in source_config.data.datasets if dataset.layout == "legacy"}
    if normalized_target and legacy_modalities:
        legacy_source = source.loc[source["Dataset"].isin(legacy_modalities)]
        confirmed_animals = set(source_config.data.animal_ids.values())
        if not set(legacy_source["Subject"]).issubset(confirmed_animals):
            raise ValueError("Legacy pretraining sources need explicit animal_ids links to the normalized inventory")
    if settings.mode == "backbone":
        fields = ("architecture", "filters", "residual", "attention", "max_pooling")
        if source_config.model.architecture == "3drnet":
            fields = (*fields, "multiscale")
        source_graph = tuple(getattr(source_config.model, name) for name in fields)
        target_graph = tuple(getattr(data.config.model, name) for name in fields)
        if source_graph != target_graph:
            raise ValueError("Source and target backbone settings differ; use a variant-matched TBI source")
    source_fold = record.fold
    parent_settings = source_config.initialization_by_fold.get(source_fold, source_config.initialization)
    # Audit ancestors against their immediate target architecture, too.
    parent_config = replace(data.config, model=source_config.model)
    parent_data = replace(data, config=parent_config)
    next_ancestors = (*ancestors, checkpoint)
    parent_audit = pretraining_audit(
        parent_data, parent_settings, protected_ids,
        validation_ids=validation_ids, ancestors=next_ancestors,
    )
    files = [checkpoint, split_file, manifest_file, experiment_file]
    state_file = split_file.parent / "training_state.json"
    if source_config.training.fixed_epochs and state_file.is_file():
        files.append(state_file)
    refit_file = run_dir / "refit.json"
    if refit_file.is_file():
        files.append(refit_file)
    contract_file = run_dir / "data_contract.json"
    if contract_file.is_file():
        files.append(contract_file)
    digests: dict[str, str] = {}
    for path in files:
        content = path.read_bytes()
        digests[str(path)] = hashlib.sha256(content).hexdigest()
    report = {
        "mode": settings.mode, "source_run": str(run_dir), "source_fold": source_fold,
        "used_animals": sorted(used_ids), "protected_animals": sorted(protected_ids),
        "sha256": digests, "ancestor": parent_audit,
    }
    if shared_validation:
        report["shared_validation_animals"] = sorted(shared_validation)
    return report


def pretraining_roles(
    data: PreparedData, initializations: Sequence[InitializationConfig],
    *, subject_ids: Sequence[str] | None = None,
) -> dict[str, str]:
    """Inherit matching animal roles from sources and their used ancestors.

    Source test animals retain that role. Ancestor test animals impose no
    restriction because they did not contribute to fitting or selection.
    Conflicting sources require aligned source splits rather than a compromise.
    """
    assignments: dict[str, set[str]] = {}
    ancestor_training: set[str] = set()
    ancestor_validation: set[str] = set()
    if subject_ids is None:
        subject_ids = data.subjects.subject_ids
    for initialization in initializations:
        settings = initialization
        visited: set[Path] = set()
        include_test: bool = True
        while settings.mode != "scratch":
            checkpoint = Path(settings.checkpoint).resolve()
            if checkpoint in visited:
                raise ValueError("Pretraining provenance contains an initialization cycle")
            visited.add(checkpoint)
            source = load_pretraining_source(settings, include_test=include_test)
            matches = source.matching_roles(data, subject_ids)
            for animal, roles in matches.items():
                if include_test:
                    assignments.setdefault(animal, set()).update(roles)
                else:
                    if "train" in roles:
                        ancestor_training.add(animal)
                    if "validation" in roles:
                        ancestor_validation.add(animal)
            settings = source.config.initialization_by_fold.get(source.fold, source.config.initialization)
            include_test = False

    result: dict[str, str] = {}
    for animal, roles in assignments.items():
        if len(roles) > 1:
            raise ValueError(f"Pretraining sources assign conflicting roles to {animal}; use aligned source splits")
        if roles:
            result[animal] = next(iter(roles))
    for animal in ancestor_training:
        if result.get(animal) in ("validation", "test"):
            raise ValueError(f"Pretraining ancestors fitted a source validation/test animal: {animal}")
        result[animal] = "train"
    for animal in ancestor_validation:
        if result.get(animal) == "test":
            raise ValueError(f"Pretraining ancestors selected on a source test animal: {animal}")
        result.setdefault(animal, "validation")
    return result


def cv_initializations_from_run(
    run_dir: Path, *, k: int, freeze_backbone: bool = False,
) -> dict[int, InitializationConfig]:
    """Select one committed source snapshot per target fold, without training.

    Matching animal memberships are checked by the common CV selection helper.
    A fixed-split source does not contain the required complete checkpoint set.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("CV initialization requires at least two folds")
    settings: dict[int, InitializationConfig] = {}
    for index in range(k):
        try:
            initialization = InitializationConfig.from_run(
                run_dir, fold=index, freeze_backbone=freeze_backbone,
            )
        except FileNotFoundError as error:
            raise ValueError(
                f"Source run has no committed checkpoint for fold {index}; "
                "use a completed source CV run from notebook 01 (K_FOLD_COMPATIBLE=True) or 02"
            ) from error
        settings[index] = initialization
    return settings


def backbone(model: Any, *, finest_only: bool = False) -> Any:
    """Extract decoder features, optionally using the shared finest boundary.

    Every decoder level is an ancestor of the finest level. Using that boundary
    permits complete feature transfer when only head fusion differs.
    """
    import tensorflow as tf

    names = {layer.name for layer in model.layers}
    if "r_output_1" in names:
        levels = (4, 3, 2, 1)
        if finest_only:
            levels = (1,)
        features = [model.get_layer(f"r_output_{level}").input for level in levels
                    if f"r_output_{level}" in names]
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
    features = backbone(model, finest_only=True)
    if settings.mode == "weights":
        model.load_weights(str(path))
        report["copied"] = [layer.name for layer in model.layers if layer.weights]
    else:
        source, source_report = _source_model(settings)
        report.update(source_report)
        source_features = backbone(source, finest_only=True)
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
