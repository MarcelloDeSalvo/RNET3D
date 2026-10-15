"""Inspect existing spatial attention gates without modifying model architectures."""

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


def attention_figure(
    result: AttentionResult, *, labels: nib.Nifti1Image | None = None, region: int | None = 1,
    axis: int = 2, max_slices: int = 24, opacity: float = 0.6,
) -> Any:
    """Build a native Plotly slice slider with MRI and fixed-scale gate overlays.

    Optional GT contours use original source IDs; region=None outlines all GT
    foreground. Attention is spatial and shared across classes, not a class
    probability or proof of a structure's influence on the final prediction.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if axis not in (0, 1, 2) or not 0 <= opacity <= 1 or max_slices < 1:
        raise ValueError("Choose axis 0/1/2, opacity between zero and one, and a positive slice limit")
    image: np.ndarray = result.image.get_fdata(dtype=np.float32)
    arrays: dict[str, np.ndarray] = {name: value.get_fdata(dtype=np.float32) for name, value in result.maps.items()}
    gt: np.ndarray | None = None
    if labels is not None:
        if labels.shape != image.shape or not np.allclose(labels.affine, result.image.affine):
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
    spacing: np.ndarray = nib.affines.voxel_sizes(result.image.affine)
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
                    hovertemplate="Attention weight: %{z:.3f}<extra></extra>",
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
    orientation = nib.aff2axcodes(result.image.affine)
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
        title=f"Spatial attention gates, coarse to fine · slice axis {axis} · axes {orientation}",
        coloraxis={"cmin": 0, "cmax": 1, "colorscale": "Inferno", "colorbar": {"title": "Gate weight"}},
        sliders=[{"steps": steps, "active": len(indices) // 2, "currentvalue": {"prefix": "Patch slice: "}}],
        height=400 * rows, template="plotly_white",
    )
    return figure
