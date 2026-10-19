"""Saved plots retain independent animal counts, split roles and missing GT."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.evaluation.run_plots import animal_metric_points, load_run_plot_data


def measurement(subject: str = "D1/mouse_a", scan: str = "day_1", **changes) -> dict:
    """One full-volume scan measurement, including metadata for grouping."""
    row = {
        "Subject": subject, "Scan": scan, "Dataset": "D1", "Region": "Lesion",
        "Metric": "Dice", "Value": 0.2, "Unit": "fraction", "Status": "ok",
        "Fold": 0, "DatasetRole": "holdout", "Condition": "TBI", "Modality": "T2w",
        "Head": "regions", "EvaluationStage": "full_volume",
    }
    row.update(changes)
    return row


def test_animal_points_average_predictions_before_timepoints_and_keep_strata():
    rows = [
        measurement(Value=0.2, OriginalID="original_mouse_a", Strain="c52"), measurement(Fold=1, Value=0.4),
        measurement(scan="day_2", Value=0.9),
        measurement(scan="T1", Modality="T1w", Value=0.8, OriginalID="original_mouse_a_T1"),
        measurement(subject="D3/sham", Dataset="D3", Condition="SHAM", Value=0.0),
        measurement(subject="D3/empty", Dataset="D3", Condition="SHAM", Value=1.0, Status="both_empty"),
        measurement(subject="D3/missing", Value=np.nan, Status="unavailable"),
        measurement(subject="D1/undefined", Value=np.inf, Status="prediction_empty"),
        measurement(subject="D1/selection", DatasetRole="validation", Value=0.99),
    ]
    points = animal_metric_points(pd.DataFrame(rows), metric="Dice")
    mouse = points.loc[points.Subject.eq("D1/mouse_a") & points.Modality.eq("T2w")].iloc[0]
    assert mouse.Value == pytest.approx(0.6)
    assert mouse.Scans == 2 and mouse.Folds == 2
    assert mouse.OriginalID == "original_mouse_a" and mouse.Strain == "c52"
    t1 = points.loc[points.Subject.eq("D1/mouse_a") & points.Modality.eq("T1w")].iloc[0]
    assert t1.OriginalID == "original_mouse_a_T1"
    assert t1.Strain == "unknown"
    assert len(points) == 4
    assert points.loc[points.Subject.eq("D3/sham"), "Value"].item() == 0.0
    assert points.loc[points.Subject.eq("D3/empty"), "Value"].item() == 1.0
    assert set(points.DatasetRole) == {"holdout"}


def test_load_prefers_committed_epochs_and_one_aggregate_metric_export(tmp_path):
    fold = tmp_path / "fold_0"
    fold.mkdir()
    state = {"history": {
        "loss": [0.9, 0.6], "val_loss": [1.2, 0.8],
        "brain_mask_mean_dice": [0.6, 0.8], "val_brain_mask_mean_dice": [None, 0.7],
    }}
    (fold / "training_state.json").write_text(json.dumps(state))
    pd.DataFrame({"epoch": [0, 1, 2], "loss": [0.9, 0.6, 0.3]}).to_csv(fold / "training.csv", index=False)
    for folder in (tmp_path / "regions", tmp_path / "summary"):
        folder.mkdir()
        pd.DataFrame([measurement()]).to_csv(folder / "full_volume_metrics.csv", index=False)
    manifest = pd.DataFrame([{
        "Subject": "D1/mouse_a", "Scan": "day_1", "Dataset": "D1",
        "OriginalID": "original_mouse_a", "Strain": "c52",
    }])
    manifest.to_csv(tmp_path / "subjects.csv", index=False)

    result = load_run_plot_data(tmp_path)
    assert result.history.Epoch.max() == 2
    assert result.metric_history.Epoch.max() == 2
    assert set(result.metric_history.Metric) == {"brain_mask_mean_dice", "val_brain_mask_mean_dice"}
    validation = result.metric_history.loc[result.metric_history.Metric.eq("val_brain_mask_mean_dice")]
    assert pd.isna(validation.Value.iloc[0]) and validation.Value.iloc[1] == 0.7
    assert len(result.metrics) == 1
    assert result.metrics.SourceFile.str.contains("summary", regex=False).all()
    assert result.metrics.OriginalID.item() == "original_mouse_a"
    assert result.metrics.Strain.item() == "c52"
    assert len(result.sources) == 3


def test_partial_cv_reads_fold_results_without_summary_and_does_not_relabel_validation(tmp_path):
    folder = tmp_path / "fold_2/evaluations/holdout/regions"
    folder.mkdir(parents=True)
    table = pd.DataFrame([measurement(Fold=2)])
    table = table.drop(columns=["Fold", "DatasetRole"])
    table.to_json(folder / "metrics.json", orient="table")
    history = pd.DataFrame({"loss": [0.8], "val_loss": [0.9]})
    history.to_csv(tmp_path / "fold_2/history.csv", index=False)

    result = load_run_plot_data(tmp_path)
    assert result.metrics.Fold.item() == 2
    assert result.metrics.DatasetRole.item() == "holdout"
    selection = result.metrics.assign(DatasetRole="validation")
    with pytest.raises(ValueError, match="available roles.*validation"):
        animal_metric_points(selection, metric="Dice")


def test_explicit_checkpoint_metrics_use_saved_contract_metadata(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    metric_file = tmp_path / "checkpoint_metrics.csv"
    row = measurement()
    del row["Condition"]
    del row["Modality"]
    pd.DataFrame([row]).to_csv(metric_file, index=False)
    contract = [{
        "animal": "D1/mouse_a", "scan": ["D1_T2w", "day_1"],
        "metadata": {"condition": "TBI", "modality": "T2w", "original_id": "original_mouse_a", "protocol": "c52"},
    }]
    (run / "data_contract.json").write_text(json.dumps(contract))
    result = load_run_plot_data(run, metrics_file=metric_file)
    assert result.metrics.Condition.item() == "TBI"
    assert result.metrics.Modality.item() == "T2w"
    assert result.metrics.OriginalID.item() == "original_mouse_a"
    assert result.metrics.Strain.item() == "c52"
    assert result.history.empty
    assert result.metric_history.empty

    contract[0]["metadata"]["strain"] = "declared-strain"
    (run / "data_contract.json").write_text(json.dumps(contract))
    explicit = load_run_plot_data(run, metrics_file=metric_file)
    assert explicit.metrics.Strain.item() == "declared-strain"

    contract[0]["metadata"].pop("strain")
    contract[0]["metadata"]["protocol"] = "unrelated-protocol"
    (run / "data_contract.json").write_text(json.dumps(contract))
    unrelated = load_run_plot_data(run, metrics_file=metric_file)
    assert unrelated.metrics.Strain.item() == "unknown"

    contract[0]["metadata"]["condition"] = "SHAM"
    (run / "data_contract.json").write_text(json.dumps(contract))
    pd.DataFrame([measurement()]).to_csv(metric_file, index=False)
    with pytest.raises(ValueError, match="Conflicting scan metadata for Condition"):
        load_run_plot_data(run, metrics_file=metric_file)


def test_missing_evaluation_still_allows_history_and_duplicate_measures_fail(tmp_path):
    folder = tmp_path / "fold_0"
    folder.mkdir()
    history = pd.DataFrame({
        "epoch": [0, 1, 1], "loss": [0.9, 0.8, 0.7], "regions_mean_dice": [0.4, 0.5, 0.6],
        "note": ["first", "uncommitted", "committed"],
    })
    history.to_csv(folder / "training.csv", index=False)
    result = load_run_plot_data(tmp_path)
    assert result.metrics.empty
    assert result.history.Value.to_list() == [0.9, 0.7]
    assert result.metric_history.Value.to_list() == [0.4, 0.6]
    assert result.metric_history.Metric.to_list() == ["regions_mean_dice", "regions_mean_dice"]

    pd.DataFrame([measurement(), measurement()]).to_csv(tmp_path / "full_volume_metrics.csv", index=False)
    with pytest.raises(ValueError, match="duplicate scan/fold"):
        load_run_plot_data(tmp_path)


@pytest.mark.parametrize("with_validation_metric", [True, False])
@pytest.mark.parametrize("validation_key", [None, "val_loss"])
def test_plotting_notebook_runs_and_exports_interactive_figures_and_animal_data(
    tmp_path, monkeypatch, with_validation_metric, validation_key,
):
    """Exercise native Plotly displays and real offline exports without a model."""
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda figure, **options: shown.append((figure, options)))
    run = tmp_path / "run"
    for fold in (0, 1):
        folder = run / f"fold_{fold}"
        folder.mkdir(parents=True)
        history = {
            "loss": [0.9, 0.5], "val_loss": [1.0, 0.7],
            "brain_mask_mean_dice": [0.7, 0.9], "regions_logits_mean_dice": [0.4, 0.6],
        }
        if with_validation_metric:
            history["val_regions_logits_mean_dice"] = [None, 0.5]
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    rows = [
        measurement(), measurement(subject="D1/mouse_b", Value=0.8, Fold=1),
        measurement(subject="D1/sham", Condition="SHAM", Modality="T1w", Value=1.0),
        measurement(subject="D3/healthy", Dataset="D3", Condition="SHAM", Value=0.95, Fold=1),
    ]
    for row in rows:
        row["OriginalID"] = "original_" + row["Subject"]
        row["Strain"] = "c52"
        if row["Dataset"] == "D3":
            row["Strain"] = "cd1"
    pd.DataFrame(rows).to_csv(run / "full_volume_metrics.csv", index=False)
    path = Path(__file__).resolve().parents[1] / "notebook/plot_training_results.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    context = {"run_path": run}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        code = code.replace("RUN_DIR: Path | None = None", "RUN_DIR: Path | None = run_path")
        code = code.replace("EXPORT_RESULTS: bool = False", "EXPORT_RESULTS: bool = True")
        code = code.replace("HISTORY_METRIC: str | None = None", 'HISTORY_METRIC: str | None = "regions_logits_mean_dice"')
        code = code.replace("FOLD_VALIDATION_KEY: str | None = None", f"FOLD_VALIDATION_KEY: str | None = {validation_key!r}")
        exec(compile(code, str(path), "exec"), context)

    export = context["export_dir"]
    overlay_expected = with_validation_metric or validation_key == "val_loss"
    expected_figures = ["loss_curves", "metric_curves", "metric_boxplots"]
    if overlay_expected:
        expected_figures.append("fold_validation_curves")
    assert len(shown) == len(expected_figures)
    assert all(options["renderer"] == "vscode" for _, options in shown)
    assert all(options["config"]["scrollZoom"] for _, options in shown)
    for name in expected_figures:
        assert (export / f"{name}.html").stat().st_size > 1_000_000
        payload = json.loads((export / f"{name}.json").read_text())
        assert payload["data"]
        assert isinstance(context["figures"][name], go.Figure)

    losses = context["figures"]["loss_curves"]
    assert len(losses.data) == 4
    assert all(trace.type == "scatter" for trace in losses.data)
    assert {trace.line.dash for trace in losses.data if trace.name == "val_loss"} == {"dash"}
    curves = context["figures"]["metric_curves"]
    training = [trace for trace in curves.data if trace.name == "regions_logits_mean_dice"]
    validation = [trace for trace in curves.data if trace.name == "val_regions_logits_mean_dice"]
    assert len(training) == 2
    for trace in training:
        assert list(trace.x) == [1, 2]
        assert list(trace.y) == [0.4, 0.6]
        assert trace.line.dash == "solid"
    if with_validation_metric:
        assert len(validation) == 2
        for trace in validation:
            assert pd.isna(trace.y[0]) and trace.y[1] == 0.5
            assert trace.line.dash == "dash" and not trace.connectgaps
    else:
        assert not validation
    if overlay_expected:
        overlay = context["figures"]["fold_validation_curves"]
        assert len(overlay.data) == 2
        assert {trace.name for trace in overlay.data} == {"Fold 0", "Fold 1"}
        assert all(trace.xaxis == "x" and trace.yaxis == "y" for trace in overlay.data)
        assert overlay.layout.height == 360
        for trace in overlay.data:
            assert list(trace.x) == [1, 2] and not trace.connectgaps
            if validation_key == "val_loss":
                assert list(trace.y) == [1.0, 0.7]
            else:
                assert pd.isna(trace.y[0]) and trace.y[1] == 0.5
    else:
        assert "fold_validation_curves" not in context["figures"]
    epoch_metrics = pd.read_csv(export / "metric_history.csv")
    assert epoch_metrics.Fold.nunique() == 2
    assert "regions_logits_mean_dice" in epoch_metrics.Metric.unique()
    boxes = context["figures"]["metric_boxplots"]
    assert all(trace.type == "box" and trace.boxpoints == "all" for trace in boxes.data)
    subjects = {subject for trace in boxes.data for subject in trace.hovertext}
    assert subjects == {row["Subject"] for row in rows}
    metadata = {row["Subject"]: (row["OriginalID"], row["Strain"]) for row in rows}
    for trace in boxes.data:
        assert "Original ID=" in trace.hovertemplate and "Strain=" in trace.hovertemplate
        for subject, custom in zip(trace.hovertext, trace.customdata):
            assert tuple(custom[:2]) == metadata[subject]
    assert boxes.layout.yaxis.matches is None
    assert boxes.layout.yaxis2.matches == "y"
    assert boxes.layout.yaxis3.matches is None
    assert boxes.layout.yaxis4.matches == "y3"
    points = pd.read_csv(export / "animal_metric_points.csv")
    assert len(points) == 4
    assert set(points.Condition) == {"TBI", "SHAM"}
    assert set(points.OriginalID) == {row["OriginalID"] for row in rows}
    assert set(points.Strain) == {"c52", "cd1"}
    settings = json.loads((export / "plot_settings.json").read_text())
    assert settings["metric"] == "Dice"
    assert settings["history_metric"] == "regions_logits_mean_dice"
    assert settings["plot_backend"] == "plotly"
    assert settings["formats"] == ["html", "json"]
    assert len(settings["sources"]) == 3
