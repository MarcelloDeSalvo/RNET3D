"""Audited complete loading or feature transfer across different label vocabularies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, InitializationConfig


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
