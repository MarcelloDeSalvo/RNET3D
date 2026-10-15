"""Animal-count classifier figures and their real notebook display cells."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.evaluation.classification import classification_report, classifier_confusion_figures


def scan_decisions():
    """Seven animals, fourteen scans, with an asymmetric 2x2 animal matrix."""
    rows = []
    decisions = [(0, 0)] * 3 + [(0, 1)] + [(1, 0)] * 2 + [(1, 1)]
    for animal, (reference, prediction) in enumerate(decisions):
        for scan in range(2):
            rows.append({
                "Variant": "mouse_full", "Seed": 42, "Fold": 0, "DatasetRole": "test",
                "Subject": f"animal_{animal}", "Scan": f"scan_{scan}", "Dataset": "D3",
                "Reference": reference, "Prediction": prediction,
            })
    return pd.DataFrame(rows)


def test_matrix_uses_animal_counts_and_reference_rows():
    report = classification_report(scan_decisions())
    figure = classifier_confusion_figures(report.confusion)[0]
    np.testing.assert_array_equal(figure.data[0].z, [[3, 1], [2, 1]])
    assert list(figure.data[0].x) == list(figure.data[0].y) == ["SHAM", "TBI"]
    assert figure.layout.xaxis.title.text == "Predicted condition"
    assert figure.layout.yaxis.title.text == "Reference condition"
    assert figure.layout.yaxis.autorange == "reversed"
    assert "N=7" in figure.layout.title.text and "seed 42" in figure.layout.title.text


def test_row_fractions_retain_counts_and_absent_classes_are_undefined():
    confusion = classification_report(scan_decisions()).confusion
    trace = classifier_confusion_figures(confusion, normalize=True)[0].data[0]
    np.testing.assert_allclose(trace.z, [[0.75, 0.25], [2 / 3, 1 / 3]])
    assert trace.text[0][0] == "3 (75.0%)"
    np.testing.assert_array_equal(trace.customdata, [[3, 1], [2, 1]])
    scans = scan_decisions().loc[lambda table: table.Reference.eq(1)]
    trace = classifier_confusion_figures(classification_report(scans).confusion, normalize=True)[0].data[0]
    assert np.isnan(trace.z[0]).all()
    assert trace.text[0][0] == "0 (undefined)"


def test_pooled_holdouts_are_preferred_without_summing_repeated_test_models():
    first = scan_decisions().assign(DatasetRole="holdout")
    second = first.assign(Fold=1, Subject="other_" + first.Subject)
    report = classification_report(pd.concat([first, second]))
    figures = classifier_confusion_figures(report.confusion)
    assert len(figures) == 1 and "pooled outer holdouts" in figures[0].layout.title.text
    assert np.asarray(figures[0].data[0].z).sum() == 14
    assert len(classifier_confusion_figures(report.confusion, include_folds=True)) == 3
    repeated = pd.concat([scan_decisions(), scan_decisions().assign(Fold=1)])
    figures = classifier_confusion_figures(classification_report(repeated).confusion)
    assert len(figures) == 2
    assert all(np.asarray(figure.data[0].z).sum() == 7 for figure in figures)


def test_variants_seeds_and_roles_remain_separate_and_validation_is_opt_in():
    original = scan_decisions()
    scans = pd.concat([
        original, original.assign(Variant="other"), original.assign(Seed=43),
        original.assign(DatasetRole="validation"),
    ])
    confusion = classification_report(scans).confusion
    assert len(classifier_confusion_figures(confusion)) == 3
    assert len(classifier_confusion_figures(confusion, roles=None)) == 4
    assert classifier_confusion_figures(pd.DataFrame()) == []


@pytest.mark.parametrize("error", ["missing", "duplicate", "negative", "fractional"])
def test_invalid_saved_matrices_are_rejected(error):
    confusion = classification_report(scan_decisions()).confusion
    if error == "missing":
        confusion = confusion.iloc[:-1]
    elif error == "duplicate":
        confusion = pd.concat([confusion, confusion.iloc[:1]])
    elif error == "negative":
        confusion.loc[0, "Count"] = -1
    else:
        confusion["Count"] = confusion["Count"].astype(float)
        confusion.loc[0, "Count"] = 0.5
    with pytest.raises(ValueError):
        classifier_confusion_figures(confusion)


@pytest.mark.parametrize("filename", [
    "05_mouse_full_training.ipynb", "06_mouse_full_kfold.ipynb", "07f_ablation.ipynb",
    "09_inference_evaluation.ipynb", "10_revision_results.ipynb",
])
def test_notebook_results_sections_display_real_classifier_matrices(filename, monkeypatch):
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebook" / filename).read_text(encoding="utf-8"))
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda figure, **kwargs: shown.append((figure, kwargs)))
    cell_id = "best-checkpoint-measurements-display"
    if filename.startswith("09_"):
        cell_id = "e6f9ff59"
    elif filename.startswith("10_"):
        cell_id = "classification-confusion-plot"
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)
    metrics = pd.DataFrame([{"Subject": "animal_0", "Scan": "scan_0", "Dataset": "D3",
                             "Region": "Lesion", "Metric": "Dice", "Status": "ok", "Value": 0.8}])
    result = SimpleNamespace(
        classifier_predictions=scan_decisions(), patch_metrics=pd.DataFrame(), evaluations={},
        metrics=metrics, run_dir=Path("unused"), study_dir=Path("unused"), output_dir=Path("unused"),
        manifest=pd.DataFrame(), timings=pd.DataFrame(),
    )
    namespace = {
        "display": lambda value: None, "result": result, "study": result, "evaluation": result,
        "prepared": SimpleNamespace(subjects=SimpleNamespace(manifest=lambda ids: pd.DataFrame(), subject_ids=[])),
        "classifier_results": classification_report(scan_decisions()),
        "analysis": SimpleNamespace(roles=("test",)), "EXPORT_CLASSIFICATION": False,
    }
    if filename.startswith("09_"):
        import plotly.express as px
        namespace["px"] = px
    if filename.startswith("07f_"):
        result.metrics = pd.DataFrame()
    exec(compile("".join(cell["source"]), filename, "exec"), namespace)
    matrices = [(figure, kwargs) for figure, kwargs in shown if "confusion matrix" in (figure.layout.title.text or "")]
    assert len(matrices) == 1
    assert matrices[0][1]["renderer"] == "vscode"
    assert np.asarray(matrices[0][0].data[0].z).sum() == 7


@pytest.mark.parametrize("filename", [
    "05_mouse_full_training.ipynb", "06_mouse_full_kfold.ipynb",
    "07f_ablation.ipynb", "09_inference_evaluation.ipynb",
])
def test_final_saved_results_cell_displays_matrix_without_training(filename, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda figure, **kwargs: shown.append(figure))
    scans = scan_decisions()
    scans.to_csv(tmp_path / "classifier_predictions.csv", index=False)
    metrics = scans.drop(columns=["Reference", "Prediction"]).assign(
        Region="Lesion", Head="regions", Metric="Dice", Value=0.8, Unit="fraction", Status="ok",
    )
    metrics.to_csv(tmp_path / "full_volume_metrics.csv", index=False)
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebook" / filename).read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][-1]["source"]))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "PAPER_RESULTS_DIR":
                node.value = ast.parse(f"Path({str(tmp_path)!r})", mode="eval").body
            elif node.target.id == "PAPER_BOOTSTRAP_SAMPLES":
                node.value = ast.Constant(100)
    ast.fix_missing_locations(tree)
    exec(compile(tree, filename, "exec"), {})
    assert len(shown) == 1 and np.asarray(shown[0].data[0].z).sum() == 7
