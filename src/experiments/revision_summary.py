"""Four compact manuscript tables; detailed evidence stays with each experiment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..evaluation.paper_tables import PaperTableConfig, PaperTables, build_paper_tables

if TYPE_CHECKING:
    from .revision_suite import SuitePlan


def _read(path: Path) -> pd.DataFrame:
    """Read a nonempty CSV, treating an empty export as unavailable evidence."""
    if not path.is_file() or path.stat().st_size < 5:
        return pd.DataFrame()
    return pd.read_csv(path)


def _result_file(folder: Path, name: str) -> Path:
    """Prefer the complete run summary over per-evaluator copies."""
    candidate = folder / "summary" / name
    if candidate.is_file():
        return candidate
    return folder / name


def _committed_folds(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    """Read completed folds when an interrupted CV job has no aggregate CSV yet."""
    measurements: list[pd.DataFrame] = []
    patches: list[pd.DataFrame] = []
    classifiers: list[pd.DataFrame] = []
    sources: list[Path] = []
    for marker in sorted(folder.glob("fold_*/completed.json")):
        record = json.loads(marker.read_text(encoding="utf-8"))
        fold = int(marker.parent.name.split("_")[-1])
        sources.append(marker)
        for entry in record["evaluations"]:
            path = marker.parent / entry["path"]
            table = pd.read_json(path, orient="table")
            measurements.append(table.assign(Fold=fold, DatasetRole=entry["role"]))
            sources.append(path)
            if entry.get("classifier_predictions"):
                path = marker.parent / entry["classifier_predictions"]
                classifiers.append(pd.read_json(path, orient="table"))
                sources.append(path)
        if record.get("patch_metrics"):
            path = marker.parent / record["patch_metrics"]
            patches.append(pd.read_json(path, orient="table"))
            sources.append(path)

    def concatenate(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    return concatenate(measurements), concatenate(patches), concatenate(classifiers), sources


def _main_rows(tables: PaperTables, task: str) -> pd.DataFrame:
    """Keep one row per variant/cohort/condition, with three understandable scores."""
    rows: list[dict[str, Any]] = []
    ids = ["Variant", "Seed", "DatasetRole", "Dataset", "Condition"]
    segmentation = tables.segmentation
    if segmentation.empty:
        return pd.DataFrame()
    for key, group in segmentation.groupby(ids, dropna=False):
        row = dict(zip(ids, key))
        row.update(Task=task, Animals=int(group["Animals"].max()), Folds=int(group["Folds"].max()))
        brain = group.loc[group["Head"].eq("brain_mask")]
        lesion = group.loc[group["Head"].eq("regions") & group["Region"].str.lower().eq("lesion")]
        for label, selected in (("BrainDice", brain), ("LesionDice", lesion)):
            row[label] = ""
            if not selected.empty:
                row[label] = selected.iloc[0].get("Dice", "")
        row["LesionHD95mm"] = ""
        if not lesion.empty:
            row["LesionHD95mm"] = lesion.iloc[0].get("HD95 (mm)", "")
        row["AnatomyDice"] = ""
        row["CompleteAnatomyAnimals"] = 0
        anatomy = tables.region_sets
        if not anatomy.empty:
            for column, value in zip(ids, key):
                anatomy = anatomy.loc[anatomy[column].eq(value)]
            anatomy = anatomy.loc[anatomy["Region"].eq("anatomy_only")]
        if not anatomy.empty:
            row["AnatomyDice"] = anatomy.iloc[0].get("Dice", "")
            row["CompleteAnatomyAnimals"] = int(anatomy.iloc[0]["Animals"])
        rows.append(row)
    return pd.DataFrame(rows)


def _classifier_checks(tables: PaperTables, task: str) -> list[dict[str, Any]]:
    """One classifier validation row per model; patches never inflate the count."""
    selected = tables.classification
    if selected.empty:
        return []
    if "Scope" in selected:
        scopes = set(selected["Scope"])
        if "pooled_holdout" in scopes:
            selected = selected.loc[selected["Scope"].eq("pooled_holdout")]
        else:
            selected = selected.loc[selected["Scope"].eq("fold")]
    rows = []
    for item in selected.to_dict(orient="records"):
        accuracy = item["accuracy"]
        low = item["accuracy_ci_lower"]
        high = item["accuracy_ci_upper"]
        confusion = tables.confusion
        for column in ("Variant", "Seed", "DatasetRole", "Scope", "Fold"):
            if column in confusion and column in item:
                confusion = confusion.loc[confusion[column].eq(item[column])]
        counts = {}
        for record in confusion.to_dict(orient="records"):
            name = f"GT{record['Reference']}_Pred{record['Prediction']}"
            counts[name] = int(record["Count"])
        details = (
            f"Sensitivity {item['sensitivity']:.3f} "
            f"[{item['sensitivity_ci_lower']:.3f}, {item['sensitivity_ci_upper']:.3f}]; "
            f"specificity {item['specificity']:.3f} "
            f"[{item['specificity_ci_lower']:.3f}, {item['specificity_ci_upper']:.3f}]; "
            f"TN={counts.get('GT0_Pred0', 0)}, FP={counts.get('GT0_Pred1', 0)}, "
            f"FN={counts.get('GT1_Pred0', 0)}, TP={counts.get('GT1_Pred1', 0)}"
        )
        rows.append({
            "Check": "SHAM/TBI classifier", "Task": task, "Variant": item["Variant"],
            "Seed": item["Seed"], "Animals": item["Animals"],
            "Estimate": f"{accuracy:.3f}", "CI_or_Limits": f"[{low:.3f}, {high:.3f}]",
            "Details": details,
        })
    return rows


def _paired_effects(rows: pd.DataFrame, tables: PaperTables) -> pd.DataFrame:
    """Add lesion Dice differences without hiding multiplicity or paired counts."""
    if rows.empty or tables.comparisons.empty:
        return rows
    comparisons = tables.comparisons
    selected = comparisons.loc[
        comparisons["Metric"].eq("Dice") & comparisons["Region"].str.lower().eq("lesion")
        & comparisons["Head"].eq("regions")
    ].copy()
    if selected.empty:
        return rows
    selected = selected.rename(columns={"Candidate": "Variant"})
    keys = ["Variant", "Seed", "DatasetRole", "Dataset", "Condition"]
    columns = [*keys, "MeanDifference", "CILower", "CIUpper", "HolmP", "PairedAnimals"]
    return rows.merge(selected[columns], on=keys, how="left", validate="one_to_one")


def _costs(folder: Path, variant: str, seed: int | None, fold: int | None = None) -> dict[str, Any]:
    """Summarize recorded durations only; no extra performance benchmark is run."""
    row: dict[str, Any] = {"Variant": variant, "Seed": seed}
    training = _read(folder / "summary" / "fold_timings.csv")
    if fold is not None and not training.empty:
        training = training.loc[training["fold"].eq(fold)]
    if not training.empty:
        row["Parameters"] = int(training["parameter_count"].max())
        row["MeanTrainingMinutes"] = float(training["training_seconds"].mean() / 60)
    inference = []
    folders = list(folder.glob("fold_*"))
    if fold is not None:
        folders = [folder / f"fold_{fold}"]
    for selected in folders:
        for role in ("holdout", "test"):
            for path in selected.glob(f"evaluations/{role}/*/inference_timings.csv"):
                table = _read(path)
                if not table.empty:
                    inference.append(table)
    if inference:
        timings = pd.concat(inference, ignore_index=True)
        columns = ["PreprocessingSeconds", "InferenceSeconds", "PostprocessingSeconds"]
        row["MeanVolumeSeconds"] = float(timings[columns].sum(axis=1).mean())
    return row


def _fixed_ablation_rows(table: pd.DataFrame, source_fold: int | None) -> pd.DataFrame:
    """Reuse only the chosen CV reference fold as a fixed-study baseline.

    Fixed runs save execution fold zero and role test. Normalize the reused
    reference to those identifiers for paired reporting; source files stay intact.
    """
    if table.empty or source_fold is None:
        return table
    selected = table.loc[table["Fold"].eq(source_fold) & table["DatasetRole"].eq("holdout")]
    return selected.assign(Fold=0, DatasetRole="test")


def _external_volume_checks(metrics: pd.DataFrame, output: Path, samples: int) -> list[dict[str, Any]]:
    """Report D4 agreement with/without correction, retaining invalid corrections."""
    from ..evaluation.statistics import AgreementConfig, agreement_table

    selected = metrics.loc[
        metrics["Head"].eq("regions") & metrics["Metric"].isin(("ReferenceVolume", "PredictionVolume"))
        & metrics["Label"].isin((1, 3, 13))
    ]
    pairs: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for metric, column in (("ReferenceVolume", "Manual"), ("PredictionVolume", "Automated")):
        values = selected.loc[selected["Metric"].eq(metric)].pivot(
            index=["Subject", "Scan"], columns="Label", values="Value",
        )
        values = values.reindex(columns=[1, 3, 13])
        frame = values.reset_index()[["Subject", "Scan"]]
        frame[column] = values[1].to_numpy()
        frame[column + "Corrected"] = (values[1] + values[13] - values[3]).to_numpy()
        pairs.append(frame)
    paired = pairs[0].merge(pairs[1], on=["Subject", "Scan"], validate="one_to_one")
    paired.to_csv(output / "d4_volume_pairs.csv", index=False)
    records = []
    for suffix in ("", "Corrected"):
        selected = paired[["Subject", "Manual" + suffix, "Automated" + suffix]].copy()
        selected = selected.rename(columns={"Manual" + suffix: "Manual", "Automated" + suffix: "Automated"})
        animals = selected.groupby("Subject", as_index=False)[["Manual", "Automated"]].mean()
        name = "Lesion"
        if suffix:
            name = "Lesion + ipsi ventricle - contra ventricle"
        row: dict[str, Any] = {"Check": "D4 volume agreement", "Region": name, "Animals": len(animals)}
        if len(animals) < 3 or not np.isfinite(animals[["Manual", "Automated"]]).all().all():
            row["Details"] = "Insufficient paired animals or unavailable volume annotations"
        elif (animals[["Manual", "Automated"]] < 0).any().any():
            row["Details"] = "Negative corrected estimates retained; agreement routine requires nonnegative measurements"
        else:
            config = AgreementConfig(unit="mm3", bootstrap_samples=samples)
            result = agreement_table(animals, config).iloc[0].to_dict()
            records.append({"Region": name, **result})
            row["Estimate"] = f"bias {result['bias']:.3f} mm3; MAE {result['mae']:.3f} mm3; ICC {result['icc_a1']:.3f}"
            row["CI_or_Limits"] = f"LoA [{result['loa_lower']:.3f}, {result['loa_upper']:.3f}] mm3"
            row["Details"] = "See audit/d4_agreement.csv for intervals and variance diagnostics"
        rows.append(row)
    pd.DataFrame(records).to_csv(output / "d4_agreement.csv", index=False)
    return rows


def supplementary_analyses(plan: SuitePlan) -> None:
    """Run configured annotation analyses without inventing missing reader data."""
    from ..evaluation.volumes import dataset_volume_report

    folder = plan.folder / "analysis"
    folder.mkdir(exist_ok=True)
    status: list[dict[str, str]] = []
    if plan.settings.include_annotation_volumes:
        root = plan.data["mouse_full"].config.data.root
        for name in ("D3", "D4"):
            source = root / name
            target = folder / f"annotated_{name}"
            if not source.is_dir():
                continue
            if not target.exists():
                try:
                    report = dataset_volume_report(source, spatial_unit=plan.settings.annotation_spatial_unit)
                except ValueError as error:
                    # Missing physical units are a data requirement, not a license
                    # to guess scale or suppress other volume-report failures.
                    if "Declare spatial_unit=" not in str(error):
                        raise
                    status.append({"Analysis": name + " annotated volumes", "Status": "needs_verified_units"})
                    continue
                report.export(target)
            status.append({"Analysis": name + " annotated volumes", "Status": "complete"})
    if plan.settings.inter_rater_manifest is None:
        status.append({"Analysis": "inter_rater", "Status": "not_configured; independent reader masks required"})
    else:
        from ..evaluation.inter_rater import inter_rater_report, load_annotation_pairs
        from ..evaluation.statistics import AgreementConfig

        target = folder / "inter_rater"
        if not target.exists():
            pairs = load_annotation_pairs(plan.settings.inter_rater_manifest)
            config = plan.data["mouse_full"].config
            regions = {source: config.labels.names[channel] for source, channel in config.labels.mapping.items() if source}
            agreement = AgreementConfig(unit="mm3", bootstrap_samples=plan.settings.bootstrap_samples)
            report = inter_rater_report(
                pairs, regions, spatial_unit=plan.settings.annotation_spatial_unit, agreement=agreement,
            )
            report.export(target)
        status.append({"Analysis": "inter_rater", "Status": "complete"})
    pd.DataFrame(status).to_csv(folder / "status.csv", index=False)


def export_suite_summary(suite_dir: Path, *, bootstrap_samples: int = 2000) -> Path:
    """Export a fresh four-sheet workbook from completed or partial saved results.

    Existing training files are read only. Missing experiments remain absent;
    coverage/provenance are saved beside the audit, not added as paper tables.
    """
    root = Path(suite_dir).resolve()
    contract = json.loads((root / "suite.json").read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    snapshot = root / "summary" / stamp
    audit = snapshot / "audit"
    paper = snapshot / "paper"
    audit.mkdir(parents=True, exist_ok=False)
    paper.mkdir()
    by_group: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for job in contract["jobs"]:
        if job["kind"] != "refit":
            path = root / "experiments" / job["name"] / f"seed_{job['seed']}"
            by_group.setdefault(job["group"], []).append((job, path))
    for name, folder in contract["settings"]["external_results"].items():
        group, variant = name.split("/", 1)
        job = {"variant": variant, "seed": None, "name": name}
        by_group.setdefault(group, []).append((job, Path(folder)))
    reference_fold = contract["settings"].get("reference_fold", 0)
    for group in list(by_group):
        if not group.endswith("_ablation"):
            continue
        task = group.removesuffix("_ablation")
        for job in contract["jobs"]:
            if job["group"] == task and job["variant"] == "reference":
                reference_job = {**job, "reference_fold": reference_fold}
                folder = root / "experiments" / job["name"] / f"seed_{job['seed']}"
                by_group[group].insert(0, (reference_job, folder))

    main: list[pd.DataFrame] = []
    ablations: list[pd.DataFrame] = []
    regions: list[pd.DataFrame] = []
    checks: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    coverage: list[dict[str, Any]] = []
    for group, jobs in by_group.items():
        measurements: list[pd.DataFrame] = []
        classifiers: list[pd.DataFrame] = []
        patches: list[pd.DataFrame] = []
        costs: list[dict[str, Any]] = []
        for job, folder in jobs:
            source_fold = job.get("reference_fold")
            identity: dict[str, Any] = {"Variant": job["variant"]}
            if job["seed"] is not None:
                identity["Seed"] = job["seed"]
            metric_path = _result_file(folder, "full_volume_metrics.csv")
            table = _read(metric_path)
            recovered_patch = pd.DataFrame()
            recovered_classifier = pd.DataFrame()
            recovered_sources: list[Path] = []
            if table.empty:
                table, recovered_patch, recovered_classifier, recovered_sources = _committed_folds(folder)
            table = _fixed_ablation_rows(table, source_fold)
            expected = len(contract["splits"].get(group, []))
            if group == "interpolation" or group.endswith("_ablation"):
                expected = 1
            observed = 0
            if not table.empty:
                table = table.assign(Variant=job["variant"])
                if job["seed"] is not None:
                    table = table.assign(Seed=job["seed"])
                if job["seed"] is None:
                    # A collaborator supplies predictions, not a fabricated
                    # training history. At least require the agreed outer IDs.
                    actual = set(map(tuple, table.loc[
                        table["DatasetRole"].eq("holdout"), ["Subject", "Fold"]
                    ].drop_duplicates().to_numpy()))
                    planned = contract["splits"].get(group, [])
                    expected_memberships = {
                        (animal, index)
                        for index, fold in enumerate(planned) for animal in fold["test"]
                    }
                    if actual != expected_memberships:
                        raise ValueError(f"Comparator {job['name']} does not match the shared outer-fold animal IDs")
                observed = table["Fold"].nunique()
                measurements.append(table)
                costs.append(_costs(folder, job["variant"], job["seed"], fold=source_fold))
                if not recovered_patch.empty:
                    recovered_patch = _fixed_ablation_rows(recovered_patch, source_fold)
                    patches.append(recovered_patch.assign(**identity))
                if not recovered_classifier.empty:
                    recovered_classifier = _fixed_ablation_rows(recovered_classifier, source_fold)
                    classifiers.append(recovered_classifier.assign(**identity))
                for name, destination in (("classifier_predictions.csv", classifiers), ("patch_metrics.csv", patches)):
                    if recovered_sources:
                        continue
                    path = _result_file(folder, name)
                    values = _read(path)
                    if not values.empty:
                        values = _fixed_ablation_rows(values, source_fold)
                        destination.append(values.assign(**identity))
                        sources.append({"Path": str(path), "SHA256": hashlib.sha256(path.read_bytes()).hexdigest()})
                selected_sources = recovered_sources
                if not selected_sources:
                    selected_sources = [metric_path]
                for path in selected_sources:
                    sources.append({"Path": str(path), "SHA256": hashlib.sha256(path.read_bytes()).hexdigest()})
            coverage.append({"Study": group, "Job": job["name"], "ObservedFolds": observed, "ExpectedFolds": expected})
        if not measurements:
            continue
        metrics = pd.concat(measurements, ignore_index=True)
        classifier = pd.DataFrame()
        patch = pd.DataFrame()
        if classifiers:
            classifier = pd.concat(classifiers, ignore_index=True)
        if patches:
            patch = pd.concat(patches, ignore_index=True)
        metrics.to_csv(audit / f"{group}_full_volume_metrics.csv", index=False)
        patch.to_csv(audit / f"{group}_patch_metrics.csv", index=False)
        roles = ("holdout", "test")
        reference = "reference"
        if group == "interpolation":
            roles = ("validation",)
            reference = "interpolation_order_1"
        if reference not in set(metrics["Variant"]):
            reference = None
        settings = PaperTableConfig(
            roles=roles, reference_variant=reference, bootstrap_samples=bootstrap_samples,
        )
        tables = build_paper_tables(metrics, patch_metrics=patch, classifier_predictions=classifier, settings=settings)
        for name, table in tables.tables().items():
            table.to_csv(audit / f"{group}_{name}.csv", index=False)
        task = group.removesuffix("_ablation")
        compact = _main_rows(tables, task)
        compact = _paired_effects(compact, tables)
        if group.endswith("_ablation"):
            compact = compact.assign(Mode="fixed_split", ReferenceFold=reference_fold)
        cost_table = pd.DataFrame(costs)
        if not cost_table.empty:
            cost_table = cost_table.loc[cost_table["Seed"].notna()]
        if not cost_table.empty:
            compact = compact.merge(cost_table, on=["Variant", "Seed"], how="left", validate="many_to_one")
        if group == "interpolation" or group.endswith("_ablation"):
            ablations.append(compact)
        else:
            controls = {
                job["variant"] for job in contract["jobs"]
                if job["group"] == group and job["variant"] != "reference"
            }
            main.append(compact.loc[~compact["Variant"].isin(controls)])
            if compact["Variant"].nunique() > 1:
                ablations.append(compact)
            reference_regions = tables.segmentation.loc[~tables.segmentation["Variant"].isin(controls)].copy()
            intervals = tables.animal_intervals
            if not reference_regions.empty and not intervals.empty:
                keys = ["Variant", "Seed", "DatasetRole", "Dataset", "Condition", "Head", "Region"]
                intervals = intervals.loc[intervals["Head"].ne("region_set"), [*keys, "Dice"]]
                intervals = intervals.rename(columns={"Dice": "DiceAnimal95CI"})
                reference_regions = reference_regions.merge(intervals, on=keys, how="left", validate="one_to_one")
            regions.append(reference_regions.assign(Task=group))
        checks.extend(_classifier_checks(tables, group))
        healthy = tables.healthy_lesion
        if not healthy.empty:
            if healthy["Scope"].eq("pooled_holdout").any():
                healthy = healthy.loc[healthy["Scope"].eq("pooled_holdout")]
            for item in healthy.to_dict(orient="records"):
                checks.append({
                    "Check": "SHAM lesion false positives", "Task": group,
                    "Variant": item["Variant"], "Seed": item["Seed"], "Animals": item["Animals"],
                    "Estimate": f"{item['FalsePositiveAnimalRate']:.3f}",
                    "CI_or_Limits": f"[{item['RateCILower']:.3f}, {item['RateCIUpper']:.3f}]",
                    "Details": f"Mean predicted lesion volume {item['MeanAnimalLesionVolumeMM3']:.3f} mm3",
                })

    external_path = root / "external" / "D4" / "full_volume_metrics.csv"
    external = _read(external_path)
    if not external.empty:
        settings = PaperTableConfig(roles=("external",), bootstrap_samples=bootstrap_samples)
        tables = build_paper_tables(external, settings=settings)
        main.append(_main_rows(tables, "D4"))
        regions.append(tables.segmentation.assign(Task="D4"))
        checks.extend(_external_volume_checks(external, audit, bootstrap_samples))
        sources.append({"Path": str(external_path), "SHA256": hashlib.sha256(external_path.read_bytes()).hexdigest()})
    symmetry = _read(root / "analysis" / "annotated_D3" / "volume_summary.csv")
    if not symmetry.empty:
        selected = symmetry.loc[symmetry["Measurement"].eq("IpsiContraRatio") & symmetry["Condition"].eq("SHAM")]
        for item in selected.to_dict(orient="records"):
            checks.append({
                "Check": "SHAM ventricle ratio", "Animals": item["FiniteAnimals"],
                "Estimate": f"{item['Mean']:.3f} +/- {item['SD']:.3f}",
                "Details": f"range {item['Minimum']:.3f} to {item['Maximum']:.3f}",
            })
    inter_rater = _read(root / "analysis" / "inter_rater" / "metric_summary.csv")
    if not inter_rater.empty:
        inter_rater.to_csv(audit / "inter_rater_summary.csv", index=False)
        agreement = _read(root / "analysis" / "inter_rater" / "volume_agreement.csv")
        for item in inter_rater.loc[inter_rater["Metric"].eq("Dice")].to_dict(orient="records"):
            volume = agreement.loc[agreement["Region"].eq(item["Region"]) & agreement["Dataset"].eq(item["Dataset"])]
            details = "Volume ICC unavailable"
            if not volume.empty and "icc_a1" in volume:
                details = f"Volume ICC(A,1) {volume.iloc[0]['icc_a1']:.3f}"
            checks.append({
                "Check": "Inter-rater Dice", "Region": item["Region"], "Animals": item["Animals"],
                "Estimate": f"{item['Mean']:.3f} +/- {item['SD']:.3f}", "Details": details,
            })

    def combined(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Keep empty studies out of paper tables without fabricating results."""
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    output = {
        "main_results": combined(main).dropna(axis=1, how="all"), "ablations": combined(ablations),
        "per_region": combined(regions), "validation": pd.DataFrame(checks),
    }
    is_draft = contract["mode"] == "draft"
    with pd.ExcelWriter(paper / "paper_tables.xlsx") as writer:
        for name, table in output.items():
            table.insert(0, "DRAFT", is_draft)
            table.to_csv(paper / f"{name}.csv", index=False)
            table.to_excel(writer, sheet_name=name, index=False)
    pd.DataFrame(coverage).to_csv(audit / "coverage.csv", index=False)
    pd.DataFrame(sources).to_csv(audit / "sources.csv", index=False)
    notes: str = ""
    if is_draft:
        notes = "DRAFT ONLY: all values are execution checks, not manuscript results.\n"
    notes += (
        "Four manuscript tables only. Detailed metrics, native patches, intervals and comparison tests are in ../audit.\n"
        "Main scores: equal-weight fold means +/- fold SD. Anatomy excludes lesion, brain and background, and requires complete annotation.\n"
        "SHAM/TBI, datasets and seeds stay separate. Missing GT is never scored. Interpolation uses validation results.\n"
        "Inspect ../audit/coverage.csv before claiming a completed study. No missing fold or experiment is invented.\n"
        "Ablation lesion differences are candidate minus reference; Holm correction includes the exported per-region/metric comparison family.\n"
        "Fixed ablations reuse only the chosen CV reference fold, with one-fold animal intervals and no fabricated fold SD.\n"
        "Classifier intervals use animals. D4 correction does not validate treatment-effect preservation; TOST is not supplied.\n"
        "Refit training supplies a deployment model, not another independent performance estimate.\n"
    )
    (paper / "README.txt").write_text(notes, encoding="utf-8")
    digest = hashlib.sha256((root / "suite.json").read_bytes()).hexdigest()
    (snapshot / "provenance.json").write_text(json.dumps({
        "suite_sha256": digest, "DRAFT": is_draft, "source_files": sources,
        "bootstrap_samples": bootstrap_samples,
    }, indent=2), encoding="utf-8")
    return paper
