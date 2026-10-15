"""Inspect spatial gates and classifier/segmentation Grad-CAM in trained models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import json

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from ..experiments.config import ExperimentConfig


@dataclass(frozen=True)
class AttentionResult:
    """Input patch, gate weights lifted to its grid, and native gate diagnostics."""

    image: nib.Nifti1Image
    maps: dict[str, nib.Nifti1Image]
    summary: pd.DataFrame


@dataclass(frozen=True)
class GradCAMResult:
    """Positive target attribution, normalized within one inspected patch."""

    image: nib.Nifti1Image
    heatmap: nib.Nifti1Image
    native_map: np.ndarray
    summary: pd.DataFrame


def load_attention_model(
    *, run_dir: Path | None = None, fold: int = 0, model_file: Path | None = None,
    experiment_file: Path | None = None, checkpoint_kind: Literal["weights", "model"] = "weights",
) -> tuple[Any, ExperimentConfig | None, Path]:
    """Load explicit best-run weights or a complete trusted Keras/HDF5 export.

    Run selection follows training_state.json's selected checkpoint, never latest
    modification time. Weights require their experiment.json to rebuild the exact
    fitting graph. Complete exports can be inspected without a saved experiment;
    their preprocessing must then be configured explicitly in the notebook.
    No compilation, transfer initialization, fitting or filesystem writes occur.
    """
    import tensorflow as tf
    from ..experiments.training import build_network, training_view

    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
        raise ValueError("Select a nonnegative fold index")
    if checkpoint_kind not in ("weights", "model"):
        raise ValueError("checkpoint_kind must be weights or model")
    if run_dir is not None:
        folder: Path = Path(run_dir) / f"fold_{fold}"
        if experiment_file is None:
            experiment_file = Path(run_dir) / "experiment.json"
        if model_file is None:
            model_file = folder / "model.keras"
            if checkpoint_kind == "weights":
                model_file = folder / "best.weights.h5"
                state_file: Path = folder / "training_state.json"
                if state_file.is_file():
                    state: dict[str, Any] = json.loads(state_file.read_text(encoding="utf-8"))
                    model_file = folder / state["selected_checkpoint"]
    if model_file is None or not Path(model_file).is_file():
        raise FileNotFoundError("Set RUN_DIR or MODEL_FILE to an existing checkpoint")
    source: Path = Path(model_file).resolve()
    config: ExperimentConfig | None = None
    if experiment_file is not None:
        config = ExperimentConfig.load(Path(experiment_file))

    if checkpoint_kind == "weights":
        if config is None:
            raise ValueError("Weights-only checkpoints require their saved experiment.json")
        model = build_network(config)
        model = training_view(model, config)
        model.load_weights(str(source))
    else:
        custom_objects: dict[str, Any] = {"tf": tf, "K": tf.keras.backend, "training": False}
        model = tf.keras.models.load_model(str(source), compile=False, custom_objects=custom_objects)
    return model, config, source


def attention_gate_table(model: Any) -> pd.DataFrame:
    """Find sigmoid spatial coefficients that feed skip-feature multiplication.

    Follow unary repeat/upsampling layers back from Multiply inputs. This matches
    current and legacy gates without selecting output-head/classifier sigmoids
    or relying on layer names. Coefficients are taken before channel repetition,
    at the resolution used to multiply the skip features.
    """
    import tensorflow as tf

    rows: list[dict[str, Any]] = []
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Multiply):
            continue
        for tensor in tf.nest.flatten(layer.input):
            coefficient = None
            for _ in range(5):
                source = tensor._keras_history.layer
                shape = tuple(tensor.shape)
                if len(shape) == 5 and shape[-1] == 1 and coefficient is None:
                    coefficient = source
                if isinstance(source, tf.keras.layers.Activation):
                    if source.get_config()["activation"] == "sigmoid" and len(shape) == 5 and shape[-1] == 1:
                        native_shape = tuple(coefficient.output.shape[1:4])
                        rows.append({
                            "Gate": source.name, "CoefficientLayer": coefficient.name,
                            "MultiplyLayer": layer.name, "NativeShape": native_shape,
                        })
                    break
                if not isinstance(source, (tf.keras.layers.Lambda, tf.keras.layers.UpSampling3D)):
                    break
                inputs = tf.nest.flatten(source.input)
                if len(inputs) != 1:
                    break
                tensor = inputs[0]
    if not rows:
        raise ValueError("No supported spatial attention gates found; this model may have attention disabled")
    if all(size is not None for row in rows for size in row["NativeShape"]):
        rows.sort(key=lambda row: np.prod(row["NativeShape"]))
    return pd.DataFrame(rows)


def inspect_attention(model: Any, image: nib.Nifti1Image) -> AttentionResult:
    """Forward one patch with training=False and lift gate weights for display.

    Native gate coefficients are unchanged. Linear, pixel-edge-aligned zoom only
    brings coarse maps onto the input patch for overlays; it does not change MRI,
    label masks, model weights or model predictions. No min-max normalization is
    applied to attention: its actual zero-to-one sigmoid scale is retained.
    """
    import tensorflow as tf

    if len(model.inputs) != 1 or len(model.input_shape) != 5:
        raise ValueError("Attention inspection requires one channels-last 3D model input")
    expected: tuple[int | None, ...] = tuple(model.input_shape[1:4])
    values: np.ndarray = image.get_fdata(dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Provide a finite preprocessed MRI patch of shape {expected}")
    if any(size is not None and actual != size for actual, size in zip(values.shape, expected)):
        raise ValueError(f"Provide a finite preprocessed MRI patch of shape {expected}")
    channels: int = int(model.input_shape[-1])
    batch: np.ndarray = np.repeat(values[None, ..., None], channels, axis=-1)
    gates: pd.DataFrame = attention_gate_table(model)
    outputs = [model.get_layer(name).output for name in gates["CoefficientLayer"]]
    probe = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    predictions = tf.nest.flatten(probe(batch, training=False))
    maps: dict[str, nib.Nifti1Image] = {}
    rows: list[dict[str, Any]] = []
    for metadata, prediction in zip(gates.to_dict(orient="records"), predictions):
        native: np.ndarray = prediction.numpy()[0, ..., 0].astype(np.float32)
        if not np.isfinite(native).all() or native.min() < 0 or native.max() > 1:
            raise ValueError("Attention coefficients must be finite sigmoid weights between zero and one")
        factors: tuple[float, ...] = tuple(target / size for target, size in zip(values.shape, native.shape))
        lifted: np.ndarray = zoom(native, factors, order=1, mode="nearest", prefilter=False, grid_mode=True)
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        maps[metadata["Gate"]] = nib.Nifti1Image(lifted, image.affine, header)
        rows.append({
            **metadata, "NativeShape": native.shape,
            "Min": float(native.min()), "Mean": float(native.mean()), "Max": float(native.max()),
        })
    return AttentionResult(image=image, maps=maps, summary=pd.DataFrame(rows))


def classifier_feature_table(model: Any) -> pd.DataFrame:
    """List spatial features on the classifier path; prefer its pre-pooling tensor.

    Decoder attention/features are excluded when the classifier branches from
    the encoder bottleneck. Their gradients for this decision are disconnected.
    """
    import tensorflow as tf

    if "classifier" not in model.output_names:
        raise ValueError("This model has no auxiliary classifier")
    classifier = model.get_layer("classifier")
    branch = tf.keras.Model(inputs=model.inputs, outputs=classifier.input)
    preferred: str | None = None
    rows: list[dict[str, Any]] = []
    for layer in branch.layers:
        outputs = tf.nest.flatten(layer.output)
        if len(outputs) == 1 and len(outputs[0].shape) == 5 and not isinstance(layer, tf.keras.layers.InputLayer):
            rows.append({"Layer": layer.name, "Type": type(layer).__name__, "Shape": tuple(outputs[0].shape)})
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling3D):
            preferred = layer.input._keras_history.layer.name
    if not rows:
        raise ValueError("The classifier has no supported spatial feature tensor")
    if preferred is None:
        preferred = rows[-1]["Layer"]
    table = pd.DataFrame(rows)
    table["Default"] = table["Layer"].eq(preferred)
    return table


def classifier_gradcam(
    model: Any, image: nib.Nifti1Image, *, target: Literal["predicted", "TBI", "SHAM"] = "predicted",
    feature_layer: str | None = None,
) -> GradCAMResult:
    """Compute 3D Grad-CAM for one patch's SHAM/TBI classifier log-odds.

    TBI targets the pre-sigmoid logit; SHAM targets its negative. Reconstructing
    this scalar from the final Dense inputs/weights avoids saturated sigmoid
    gradients without changing the classifier activation. Spatially averaged
    gradients weight feature channels; ReLU retains positive evidence, normalized
    by its native maximum. Zero gradients/positive maps stay zero and are flagged.
    This is neither a class probability nor a whole-volume majority-vote explanation.
    """
    import tensorflow as tf

    if target not in ("predicted", "TBI", "SHAM"):
        raise ValueError("Grad-CAM target must be predicted, TBI or SHAM")
    candidates: pd.DataFrame = classifier_feature_table(model)
    if feature_layer is None:
        feature_layer = candidates.loc[candidates["Default"], "Layer"].iloc[0]
    if feature_layer not in set(candidates["Layer"]):
        raise ValueError("Choose a spatial feature layer on the classifier path, not a disconnected decoder layer")
    if len(model.inputs) != 1 or len(model.input_shape) != 5:
        raise ValueError("Grad-CAM requires one channels-last 3D model input")
    values: np.ndarray = image.get_fdata(dtype=np.float32)
    expected: tuple[int | None, ...] = tuple(model.input_shape[1:4])
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Provide a finite 3D preprocessed MRI patch")
    if any(size is not None and size != actual for size, actual in zip(expected, values.shape)):
        raise ValueError(f"Provide a preprocessed MRI patch of shape {expected}")
    classifier = model.get_layer("classifier")
    dense_sigmoid: bool = isinstance(classifier, tf.keras.layers.Dense) and classifier.units == 1
    activation_sigmoid: bool = isinstance(classifier, tf.keras.layers.Activation) and classifier.input.shape[-1] == 1
    scalar_output: bool = len(classifier.output.shape) == 2 and classifier.output.shape[-1] == 1
    if not scalar_output or not (dense_sigmoid or activation_sigmoid) or classifier.get_config()["activation"] != "sigmoid":
        raise ValueError("Grad-CAM supports a single sigmoid classifier output")
    features = model.get_layer(feature_layer).output
    probe = tf.keras.Model(inputs=model.inputs, outputs=[features, classifier.input])
    channels: int = int(model.input_shape[-1])
    batch: np.ndarray = np.repeat(values[None, ..., None], channels, axis=-1)
    inputs = tf.convert_to_tensor(batch)
    resolved_target: str = target
    with tf.GradientTape() as tape:
        # Watch input explicitly so frozen backbones also retain this derivative.
        tape.watch(inputs)
        activations, head_inputs = probe(inputs, training=False)
        logits = tf.cast(head_inputs, tf.float32)
        if dense_sigmoid:
            kernel = tf.cast(classifier.kernel, tf.float32)
            logits = tf.linalg.matmul(logits, kernel)
            if classifier.use_bias:
                logits = logits + tf.cast(classifier.bias, tf.float32)
        logit = logits[0, 0]
        probability = tf.math.sigmoid(logit)
        if target == "predicted":
            resolved_target = "SHAM"
            if float(probability.numpy()) > 0.5:
                resolved_target = "TBI"
        score = logit
        if resolved_target == "SHAM":
            score = -logit
    gradients = tape.gradient(score, activations)
    if gradients is None:
        raise ValueError("The selected feature tensor has no gradient connection to the classifier")
    metadata = {
        "Target": resolved_target, "FeatureLayer": feature_layer,
        "TBIPatchProbability": float(probability.numpy()), "TBILogit": float(logit.numpy()),
    }
    return _gradcam_result(image, activations, gradients, score, metadata)


def _gradcam_result(image: nib.Nifti1Image, activations: Any, gradients: Any, score: Any,
                    metadata: dict[str, Any]) -> GradCAMResult:
    """Apply shared 3D Grad-CAM reduction and retain its unnormalized diagnostics."""
    import tensorflow as tf

    gradients = tf.cast(gradients[0], tf.float32)
    activations = tf.cast(activations[0], tf.float32)
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    signed = tf.reduce_sum(activations * weights, axis=-1)
    raw: np.ndarray = signed.numpy()
    gradient_values: np.ndarray = gradients.numpy()
    if not np.isfinite(raw).all() or not np.isfinite(gradient_values).all() or not np.isfinite(float(score.numpy())):
        raise ValueError("Grad-CAM requires finite scores, features and gradients")
    native: np.ndarray = np.maximum(raw, 0)
    peak: float = float(native.max())
    status: str = "no_positive_attribution"
    if peak > 0:
        native = native / peak
        status = "ok"
    if not np.any(gradient_values):
        status = "zero_gradient"
    factors: tuple[float, ...] = tuple(size / coarse for size, coarse in zip(image.shape, native.shape))
    lifted: np.ndarray = zoom(native, factors, order=1, mode="nearest", prefilter=False, grid_mode=True)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    heatmap = nib.Nifti1Image(lifted, image.affine, header)
    summary = pd.DataFrame([{
        **metadata, "NativeShape": native.shape,
        "TargetScore": float(score.numpy()), "GradientL1": float(np.abs(gradient_values).sum()),
        "RawMinimum": float(raw.min()), "RawMaximum": float(raw.max()), "PositivePeak": peak,
        "Status": status,
    }])
    return GradCAMResult(image=image, heatmap=heatmap, native_map=native, summary=summary)


def _segmentation_logits(model: Any, head: str) -> tuple[Any, str]:
    """Expose existing raw segmentation scores without changing activations."""
    import tensorflow as tf

    name: str = head
    if head == "regions" and "regions_logits" in model.output_names:
        name = "regions_logits"
    if name not in model.output_names:
        raise ValueError(f"The model has no {head} output")
    layer = model.get_layer(name)
    if not isinstance(layer, tf.keras.layers.Activation):
        raise ValueError("Supported segmentation heads expose their final activation as a separate layer")
    activation: str = layer.get_config()["activation"]
    if activation in ("softmax", "sigmoid"):
        return layer.input, name
    if name == "regions_logits" and activation == "linear":
        return layer.output, name
    raise ValueError("Expected a softmax/sigmoid segmentation head or linear regions_logits")


def segmentation_feature_table(model: Any, *, head: Literal["regions", "brain_mask"] = "regions") -> pd.DataFrame:
    """List spatial ancestors of a segmentation head, preferring finest decoder features."""
    import tensorflow as tf

    logits, _ = _segmentation_logits(model, head)
    branch = tf.keras.Model(inputs=model.inputs, outputs=logits)
    preferred: str | None = None
    projection: str = "r_output_1"
    if head == "brain_mask":
        projection = "b_output_1"
    names = {layer.name for layer in branch.layers}
    if projection in names:
        preferred = branch.get_layer(projection).input._keras_history.layer.name
    else:
        producer = logits._keras_history.layer
        if isinstance(producer, tf.keras.layers.Activation):
            producer = producer.input._keras_history.layer
        if isinstance(producer, tf.keras.layers.Conv3D):
            preferred = producer.input._keras_history.layer.name
    rows: list[dict[str, Any]] = []
    for layer in branch.layers:
        tensors = tf.nest.flatten(layer.output)
        if len(tensors) == 1 and len(tensors[0].shape) == 5 and not isinstance(layer, tf.keras.layers.InputLayer):
            rows.append({"Layer": layer.name, "Type": type(layer).__name__, "Shape": tuple(tensors[0].shape)})
    if not rows:
        raise ValueError("No spatial features found on this segmentation path")
    table = pd.DataFrame(rows)
    if preferred not in set(table["Layer"]):
        preferred = rows[-1]["Layer"]
    table["Default"] = table["Layer"].eq(preferred)
    return table


def segmentation_gradcam(
    model: Any, image: nib.Nifti1Image, *, class_index: int = 1,
    head: Literal["regions", "brain_mask"] = "regions",
    mask_mode: Literal["all", "prediction", "provided"] = "all",
    spatial_mask: nib.Nifti1Image | None = None, feature_layer: str | None = None,
) -> GradCAMResult:
    """Explain a selected class's mean raw logit over fixed target voxels.

    all targets the whole patch. prediction fixes voxels to the class argmax
    (positive sigmoid brain for brain_mask). provided requires an aligned binary
    mask whose annotation validity is established by the caller. Empty targets
    are rejected; missing GT is never interpreted as known absence. No softmax
    probabilities are summed across classes. Models without a classifier work.
    """
    import tensorflow as tf

    if head not in ("regions", "brain_mask") or mask_mode not in ("all", "prediction", "provided"):
        raise ValueError("Choose a supported segmentation head and all/prediction/provided target voxels")
    logits_tensor, output_name = _segmentation_logits(model, head)
    classes: int = int(logits_tensor.shape[-1])
    if isinstance(class_index, bool) or not isinstance(class_index, int) or not 0 <= class_index < classes:
        raise ValueError(f"class_index must be a model channel in [0, {classes}); brain_mask uses channel 0")
    candidates = segmentation_feature_table(model, head=head)
    if feature_layer is None:
        feature_layer = candidates.loc[candidates["Default"], "Layer"].iloc[0]
    if feature_layer not in set(candidates["Layer"]):
        raise ValueError("Select a spatial feature on the chosen segmentation path")
    values = image.get_fdata(dtype=np.float32)
    if len(model.inputs) != 1 or values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Provide a finite 3D MRI patch for a single-input model")
    expected = model.input_shape[1:4]
    if any(size is not None and size != actual for size, actual in zip(expected, image.shape)):
        raise ValueError(f"Expected a preprocessed MRI patch of shape {expected}")
    provided: np.ndarray | None = None
    if mask_mode == "provided":
        if spatial_mask is None or spatial_mask.shape != image.shape or not np.allclose(spatial_mask.affine, image.affine, rtol=0, atol=1e-6):
            raise ValueError("Provide a binary target mask sharing the MRI patch grid")
        provided = spatial_mask.get_fdata()
        if not np.isin(provided, [0, 1]).all():
            raise ValueError("The provided target mask must be finite and binary")
    elif spatial_mask is not None:
        raise ValueError("A supplied mask requires mask_mode=provided")
    channels = int(model.input_shape[-1])
    batch = np.repeat(values[None, ..., None], channels, axis=-1)
    inputs = tf.convert_to_tensor(batch)
    feature_tensor = model.get_layer(feature_layer).output
    probe = tf.keras.Model(inputs=model.inputs, outputs=[feature_tensor, logits_tensor])
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        activations, logits = probe(inputs, training=False)
        logits = tf.cast(logits[0], tf.float32)
        target_mask = np.ones(image.shape, dtype=np.float32)
        if mask_mode == "provided":
            target_mask = provided.astype(np.float32)
        elif mask_mode == "prediction":
            if head == "brain_mask":
                target_mask = (logits.numpy()[..., 0] > 0).astype(np.float32)
            else:
                target_mask = (np.argmax(logits.numpy(), axis=-1) == class_index).astype(np.float32)
        voxel_count = int(target_mask.sum())
        if voxel_count == 0:
            raise ValueError("No target voxels: choose all, a nonempty prediction, or a genuinely annotated nonempty mask")
        fixed_mask = tf.stop_gradient(tf.convert_to_tensor(target_mask))
        score = tf.reduce_sum(logits[..., class_index] * fixed_mask) / voxel_count
    gradients = tape.gradient(score, activations)
    if gradients is None:
        raise ValueError("Selected features are disconnected from this segmentation target")
    metadata = {
        "Target": f"{head} channel {class_index}", "Head": head, "Output": output_name,
        "ClassIndex": class_index, "FeatureLayer": feature_layer,
        "MaskMode": mask_mode, "TargetVoxels": voxel_count, "ScoreReduction": "mean_logit",
    }
    return _gradcam_result(image, activations, gradients, score, metadata)


def _spatial_maps_figure(
    source_image: nib.Nifti1Image, maps: dict[str, nib.Nifti1Image], *,
    title: str, colorbar_title: str, hover_template: str,
    labels: nib.Nifti1Image | None = None, region: int | None = 1,
    axis: int = 2, max_slices: int = 24, opacity: float = 0.6,
) -> Any:
    """Build aligned native Plotly slices with explicit map meaning and scale."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if axis not in (0, 1, 2) or not 0 <= opacity <= 1 or max_slices < 1:
        raise ValueError("Choose axis 0/1/2, opacity between zero and one, and a positive slice limit")
    image: np.ndarray = source_image.get_fdata(dtype=np.float32)
    arrays: dict[str, np.ndarray] = {name: value.get_fdata(dtype=np.float32) for name, value in maps.items()}
    gt: np.ndarray | None = None
    if labels is not None:
        if labels.shape != image.shape or not np.allclose(labels.affine, source_image.affine):
            raise ValueError("GT contours must share the MRI patch grid")
        gt = labels.get_fdata()
        if region is None:
            gt = gt > 0
        else:
            gt = gt == region
    names: list[str] = ["MRI", *arrays]
    columns: int = min(3, len(names))
    rows: int = int(np.ceil(len(names) / columns))
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=names)
    horizontal, vertical = [dimension for dimension in range(3) if dimension != axis]
    spacing: np.ndarray = nib.affines.voxel_sizes(source_image.affine)
    x: np.ndarray = np.arange(image.shape[horizontal]) * spacing[horizontal]
    y: np.ndarray = np.arange(image.shape[vertical]) * spacing[vertical]
    low, high = np.percentile(image, [1, 99])
    high = max(float(high), float(low) + 1e-6)
    count: int = min(max_slices, image.shape[axis])
    indices: np.ndarray = np.unique(np.linspace(0, image.shape[axis] - 1, count, dtype=int))

    def traces(index: int) -> list[Any]:
        """Use the same MRI slice, coefficient grid and GT contour in all panels."""
        items: list[Any] = []
        pixels: np.ndarray = np.take(image, index, axis=axis).T
        for name in names:
            items.append(go.Heatmap(z=pixels, x=x, y=y, zmin=low, zmax=high, colorscale="Gray", showscale=False))
            if name != "MRI":
                weights: np.ndarray = np.take(arrays[name], index, axis=axis).T
                items.append(go.Heatmap(
                    z=weights, x=x, y=y, coloraxis="coloraxis", opacity=opacity,
                    hovertemplate=hover_template,
                ))
            if gt is not None:
                contour: np.ndarray = np.take(gt, index, axis=axis).T.astype(int)
                items.append(go.Contour(
                    z=contour, x=x, y=y, showscale=False, hoverinfo="skip",
                    contours={"start": 0.5, "end": 0.5, "size": 1, "coloring": "none"},
                    line={"color": "cyan", "width": 1.5},
                ))
        return items

    initial: int = int(indices[len(indices) // 2])
    initial_traces = iter(traces(initial))
    orientation = nib.aff2axcodes(source_image.affine)
    for panel, name in enumerate(names):
        row, column = divmod(panel, columns)
        trace_count: int = 1
        if name != "MRI":
            trace_count += 1
        if gt is not None:
            trace_count += 1
        for _ in range(trace_count):
            figure.add_trace(next(initial_traces), row=row + 1, col=column + 1)
        anchor: str = "x"
        if panel:
            anchor = f"x{panel + 1}"
        figure.update_xaxes(title_text=f"Axis {horizontal} toward {orientation[horizontal]} (mm)", row=row + 1, col=column + 1)
        figure.update_yaxes(
            title_text=f"Axis {vertical} toward {orientation[vertical]} (mm)",
            scaleanchor=anchor, autorange="reversed", row=row + 1, col=column + 1,
        )
    figure.frames = [go.Frame(name=str(index), data=traces(int(index))) for index in indices]
    steps = [{"label": str(index), "method": "animate", "args": [[str(index)], {
        "mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0},
    }]} for index in indices]
    figure.update_layout(
        title=f"{title} · slice axis {axis} · axes {orientation}",
        coloraxis={"cmin": 0, "cmax": 1, "colorscale": "Inferno", "colorbar": {"title": colorbar_title}},
        sliders=[{"steps": steps, "active": len(indices) // 2, "currentvalue": {"prefix": "Patch slice: "}}],
        height=400 * rows, template="plotly_white",
    )
    return figure


def attention_figure(
    result: AttentionResult, *, labels: nib.Nifti1Image | None = None, region: int | None = 1,
    axis: int = 2, max_slices: int = 24, opacity: float = 0.6,
) -> Any:
    """Show class-shared gate weights on their actual zero-to-one sigmoid scale."""
    title: str = "Spatial attention gates, coarse to fine"
    hover: str = "Attention weight: %{z:.3f}<extra></extra>"
    return _spatial_maps_figure(
        source_image=result.image, maps=result.maps, title=title,
        colorbar_title="Gate weight", hover_template=hover, labels=labels, region=region,
        axis=axis, max_slices=max_slices, opacity=opacity,
    )


def gradcam_figure(
    result: GradCAMResult, *, labels: nib.Nifti1Image | None = None, region: int | None = 1,
    axis: int = 2, max_slices: int = 24, opacity: float = 0.6,
) -> Any:
    """Show normalized positive classifier/segmentation attribution and its target."""
    metadata = result.summary.iloc[0]
    name: str = f"Grad-CAM for {metadata['Target']}"
    maps: dict[str, nib.Nifti1Image] = {name: result.heatmap}
    title: str = f"{name} · {metadata['FeatureLayer']} · {metadata['Status']}"
    hover: str = "Relative positive attribution: %{z:.3f}<extra></extra>"
    return _spatial_maps_figure(
        source_image=result.image, maps=maps, title=title,
        colorbar_title="Normalized attribution", hover_template=hover, labels=labels, region=region,
        axis=axis, max_slices=max_slices, opacity=opacity,
    )
