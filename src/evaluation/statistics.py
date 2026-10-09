"""Agreement statistics for paired manual and automated measurements.

Implements Bland–Altman bias and 95% limits of agreement (LoA),
absolute-agreement single-measure ICC(A,1), Lin's CCC and MAE. Adds paired
tests, explicitly labelled variance diagnostics and a separate binary
classification helper with independent-subject confidence intervals.

Each row must represent one independent subject with one measurement from
both methods. Longitudinal scans or multiple fold-model predictions require
an explicitly defined aggregation or hierarchical analysis beforehand.

Dependencies: numpy, pandas, scipy; plotly for optional interactive figures.
Example::

    python -m evaluation.statistics paired_volumes.csv --output agreement_report \
        --subject Subject --reference Manual --prediction Automated --unit mm3
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, levene, linregress, probplot, shapiro, ttest_rel, wilcoxon


@dataclass(frozen=True, kw_only=True)
class AgreementConfig:
    """Analysis settings.

    Attributes:
        subject_col: Unique identifier for each independent biological subject.
        reference_col: Column containing manual/reference measurements.
        prediction_col: Column containing automated measurements.
        group_cols: Optional columns for separate subgroup reports. Group reports
            are descriptive and do not constitute multiple independent studies.
        unit: Display unit; no unit conversion is performed.
        confidence: Bootstrap confidence level, between zero and one.
        bootstrap_samples: Number of subject-level bootstrap replicates (>= 100).
        seed: Random seed for reproducible uncertainty estimates.
        loa_z: Standard-normal multiplier for classical Bland–Altman LoA.
            Default uses a central 95% normal reference distribution.
        include_paired_t: Add an optional paired t-test; never selected based on
            the normality-test p-value. Paired bootstrap intervals are always
            computed for the descriptive SD reduction and proportional slope.
    """

    subject_col: str = "Subject"
    reference_col: str = "Manual"
    prediction_col: str = "Automated"
    group_cols: tuple[str, ...] = ()
    unit: str = ""
    confidence: float = 0.95
    bootstrap_samples: int = 2000
    seed: int = 42
    loa_z: float = 1.959963984540054
    include_paired_t: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie strictly between 0 and 1")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be >= 100")
        if self.loa_z <= 0:
            raise ValueError("loa_z must be positive")


@dataclass(frozen=True)
class AgreementResult:
    """Point estimates and percentile bootstrap CIs for one cohort.

    Attributes:
        n: Number of independent paired subjects.
        bias: Mean (automated - manual) difference.
        sd_difference: Sample standard deviation of the paired differences.
        loa_lower: Lower normal-theory Bland–Altman limit of agreement.
        loa_upper: Upper normal-theory Bland–Altman limit of agreement.
        mae: Mean absolute error.
        ccc: Lin's concordance correlation coefficient.
        icc_a1: Two-way absolute-agreement, single-measure ICC(A,1).
        ci: Mapping from statistic to two-element bootstrap percentile interval;
            undefined intervals are represented by None.
        sd_reduction_percent: 100 * (1 - prediction SD / reference SD), with
            paired-subject bootstrap bounds in ci. Undefined for zero reference SD.
        proportional_slope: OLS slope of differences versus pair means.
        diagnostics: Paired tests, explicitly unpaired Levene, and difference
            normality diagnostics. P-values are exploratory and unadjusted.
    """

    n: int
    bias: float
    sd_difference: float
    loa_lower: float
    loa_upper: float
    mae: float
    ccc: float
    icc_a1: float
    ci: dict[str, tuple[float, float] | None]
    rmse: float = float('nan')
    pearson_r: float = float('nan')
    reference_sd: float = float('nan')
    prediction_sd: float = float('nan')
    sd_reduction_percent: float = float('nan')
    proportional_slope: float = float('nan')
    diagnostics: dict[str, float | str] = field(default_factory=dict)


def _validate_pairs(frame: pd.DataFrame, config: AgreementConfig) -> pd.DataFrame:
    """Return validated paired records without duplicating or dropping subjects.

    Raises ValueError for absent data, repeated IDs, NaN, infinite or negative
    values (volume measurements are assumed nonnegative).
    """
    cols = [config.subject_col, config.reference_col, config.prediction_col, *config.group_cols]
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = frame.loc[:, cols].copy()
    if data.empty or data.isna().any().any():
        raise ValueError("Paired measurements must be nonempty and contain no missing values")
    if data[config.subject_col].duplicated().any():
        raise ValueError("Duplicate subject identifiers: aggregate repeated scans/folds explicitly first")
    for col in (config.reference_col, config.prediction_col):
        data[col] = pd.to_numeric(data[col], errors="raise")
        values = data[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{col} contains non-finite values")
        if (values < 0).any():
            raise ValueError(f"{col} contains negative volume measurements")
    return data


def _estimates(reference: np.ndarray, prediction: np.ndarray, z: float) -> dict[str, float]:
    """Calculate agreement statistics on paired arrays, without uncertainty."""
    n = reference.size
    differences = prediction - reference
    bias = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else float("nan")

    mean_r, mean_p = np.mean(reference), np.mean(prediction)
    var_r, var_p = np.var(reference), np.var(prediction)
    covariance = np.mean((reference - mean_r) * (prediction - mean_p))
    ccc_denominator = var_r + var_p + (mean_r - mean_p) ** 2
    ccc = 2 * covariance / ccc_denominator if ccc_denominator > 0 else float("nan")

    # Two-way mixed absolute-agreement ICC(A,1), k=2 fixed raters.
    measurements = np.column_stack((reference, prediction))
    k = 2
    grand = measurements.mean()
    rows = measurements.mean(axis=1)
    columns = measurements.mean(axis=0)
    ss_subject = k * np.sum((rows - grand) ** 2)
    ss_rater = n * np.sum((columns - grand) ** 2)
    ss_error = np.sum((measurements - rows[:, None] - columns[None, :] + grand) ** 2)
    ms_subject = ss_subject / (n - 1) if n > 1 else np.nan
    ms_rater = ss_rater / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if n > 1 else np.nan
    icc_denominator = ms_subject + (k - 1) * ms_error + k * (ms_rater - ms_error) / n
    icc = (ms_subject - ms_error) / icc_denominator if icc_denominator > 0 else np.nan

    reference_sd: float = float(np.std(reference, ddof=1))
    prediction_sd: float = float(np.std(prediction, ddof=1))
    reduction: float = float('nan')
    pearson_r: float = float('nan')
    slope: float = float('nan')
    if reference_sd > 0:
        reduction = 100 * (1 - prediction_sd / reference_sd)
    if var_r > 0 and var_p > 0:
        pearson_r = float(covariance / np.sqrt(var_r * var_p))

    means: np.ndarray = (reference + prediction) / 2
    centred_means: np.ndarray = means - means.mean()
    mean_square: float = float(np.sum(centred_means ** 2))
    if mean_square > 0:
        slope = float(np.sum(centred_means * (differences - bias)) / mean_square)

    return {
        "bias": bias,
        "sd_difference": sd,
        "loa_lower": bias - z * sd,
        "loa_upper": bias + z * sd,
        "mae": float(np.mean(np.abs(differences))),
        "ccc": float(ccc),
        "icc_a1": float(icc),
        "rmse": float(np.sqrt(np.mean(differences ** 2))),
        "pearson_r": pearson_r,
        "reference_sd": reference_sd,
        "prediction_sd": prediction_sd,
        "sd_reduction_percent": reduction,
        "proportional_slope": slope,
    }


def _paired_diagnostics(
    reference: np.ndarray, prediction: np.ndarray, include_paired_t: bool,
) -> dict[str, float | str]:
    """Separate paired location, independent-sample variance and bias diagnostics.

    Wilcoxon uses SciPy's auto method with zero differences discarded. Its
    approximation with ties/zeros depends on the installed SciPy version.
    The all-zero case is explicitly reported as statistic=0, p=1. Levene's
    median-centred test ignores pairing; paired bootstrap SD-reduction CIs in
    AgreementResult provide the companion sensitivity analysis, not a Levene p.
    """
    differences: np.ndarray = prediction - reference
    diagnostics: dict[str, float | str] = {
        'wilcoxon_method': 'auto; two-sided; zero_method=wilcox',
        'levene_assumption': 'independent samples; does not account for paired animals',
        'variance_sensitivity': 'paired-subject percentile bootstrap of SD reduction',
        'proportional_bias_note': 'OLS difference versus pair mean; diagnostic, not proof of mechanism',
    }
    if np.all(differences == 0):
        diagnostics['wilcoxon_statistic'] = 0.0
        diagnostics['wilcoxon_pvalue'] = 1.0
    else:
        paired = wilcoxon(differences, zero_method='wilcox', alternative='two-sided', method='auto')
        diagnostics['wilcoxon_statistic'] = float(paired.statistic)
        diagnostics['wilcoxon_pvalue'] = float(paired.pvalue)

    deviations_r: np.ndarray = np.abs(reference - np.median(reference))
    deviations_p: np.ndarray = np.abs(prediction - np.median(prediction))
    within_variation: float = float(np.var(deviations_r) + np.var(deviations_p))
    diagnostics['levene_statistic'] = float('nan')
    diagnostics['levene_pvalue'] = float('nan')
    if within_variation > 0:
        variance_test = levene(reference, prediction, center='median')
        diagnostics['levene_statistic'] = float(variance_test.statistic)
        diagnostics['levene_pvalue'] = float(variance_test.pvalue)

    means: np.ndarray = (reference + prediction) / 2
    diagnostics['proportional_slope_pvalue'] = float('nan')
    diagnostics['proportional_intercept'] = float('nan')
    if np.ptp(means) > 0 and np.ptp(differences) > 0:
        regression = linregress(means, differences)
        diagnostics['proportional_slope_pvalue'] = float(regression.pvalue)
        diagnostics['proportional_intercept'] = float(regression.intercept)

    diagnostics['difference_shapiro_statistic'] = float('nan')
    diagnostics['difference_shapiro_pvalue'] = float('nan')
    if 3 <= len(differences) <= 5000 and np.ptp(differences) > 0:
        normality = shapiro(differences)
        diagnostics['difference_shapiro_statistic'] = float(normality.statistic)
        diagnostics['difference_shapiro_pvalue'] = float(normality.pvalue)

    if include_paired_t:
        diagnostics['paired_t_statistic'] = float('nan')
        diagnostics['paired_t_pvalue'] = float('nan')
        if np.ptp(differences) > 0:
            paired_t = ttest_rel(prediction, reference)
            diagnostics['paired_t_statistic'] = float(paired_t.statistic)
            diagnostics['paired_t_pvalue'] = float(paired_t.pvalue)

    return diagnostics


def calculate_agreement(frame: pd.DataFrame, config: AgreementConfig = AgreementConfig()) -> AgreementResult:
    """Compute agreement, subject-level bootstrap confidence intervals.

    Args:
        frame: One row per independent subject; manual and automated columns.
        config: Data mapping, uncertainty, and reproducibility settings.

    Returns:
        Point estimates and percentile bootstrap intervals. ICC(A,1) is a
        single-measure, absolute-agreement ICC for two fixed methods.

    Notes:
        Normal-theory Bland–Altman LoA presume approximately normal paired
        differences and homogeneous difference variance. Bootstrap intervals
        quantify sampling uncertainty, not the expected range of individual
        differences. Interpret LoA cautiously for small or nonnormal cohorts.
    """
    data = _validate_pairs(frame, config)
    if len(data) < 3:
        raise ValueError("At least 3 independent paired subjects are needed")
    reference = data[config.reference_col].to_numpy(dtype=float)
    prediction = data[config.prediction_col].to_numpy(dtype=float)
    point = _estimates(reference, prediction, config.loa_z)

    rng = np.random.default_rng(config.seed)
    bootstrap = {name: [] for name in point}
    for _ in range(config.bootstrap_samples):
        sample = rng.integers(0, len(data), len(data))
        estimates = _estimates(reference[sample], prediction[sample], config.loa_z)
        for name, value in estimates.items():
            if np.isfinite(value):
                bootstrap[name].append(value)

    alpha = (1 - config.confidence) / 2
    ci = {
        name: (float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha)))
        if np.isfinite(point[name]) and len(values) >= 0.8 * config.bootstrap_samples
        else None
        for name, values in bootstrap.items()
    }
    diagnostics = _paired_diagnostics(reference, prediction, config.include_paired_t)
    return AgreementResult(n=len(data), **point, ci=ci, diagnostics=diagnostics)


def agreement_table(frame: pd.DataFrame, config: AgreementConfig = AgreementConfig()) -> pd.DataFrame:
    """Return one CSV-friendly summary row per cohort or optional subgroup."""
    data = _validate_pairs(frame, config)
    groups = [((), data)] if not config.group_cols else list(data.groupby(list(config.group_cols), dropna=False))
    rows = []
    for group_key, subgroup in groups:
        result = calculate_agreement(subgroup, config)
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(config.group_cols, keys))
        row.update({"n": result.n, "unit": config.unit})
        for metric in result.ci:
            row[metric] = getattr(result, metric)
            bounds = result.ci[metric]
            row[f"{metric}_ci_lower"] = bounds[0] if bounds else np.nan
            row[f"{metric}_ci_upper"] = bounds[1] if bounds else np.nan
        row.update(result.diagnostics)
        rows.append(row)
    return pd.DataFrame(rows)


def bland_altman_figure(frame: pd.DataFrame, result: AgreementResult, config: AgreementConfig = AgreementConfig()):
    """Build an interactive Bland–Altman Plotly figure; Plotly is optional."""
    import plotly.graph_objects as go

    data = _validate_pairs(frame, config)
    manual = data[config.reference_col].to_numpy(dtype=float)
    automated = data[config.prediction_col].to_numpy(dtype=float)
    means = (manual + automated) / 2
    differences = automated - manual
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=means, y=differences, mode="markers", name="Subjects",
        text=data[config.subject_col].astype(str), hovertemplate="%{text}<br>Mean: %{x:.3f}<br>Difference: %{y:.3f}<extra></extra>",
    ))
    for value, name, dash in (
        (result.bias, "Mean bias", "solid"),
        (result.loa_lower, "Lower LoA", "dash"),
        (result.loa_upper, "Upper LoA", "dash"),
    ):
        fig.add_hline(y=value, line_dash=dash, annotation_text=f"{name}: {value:.3f}")
    if np.isfinite(result.proportional_slope):
        line_x: np.ndarray = np.array([means.min(), means.max()])
        line_y: np.ndarray = result.bias + result.proportional_slope * (line_x - means.mean())
        trend = go.Scatter(x=line_x, y=line_y, mode='lines', name='Proportional-bias diagnostic')
        fig.add_trace(trend)
    suffix = f" ({config.unit})" if config.unit else ""
    fig.update_layout(
        title=f"Bland–Altman agreement (n={result.n})",
        xaxis_title=f"Mean of manual and automated{suffix}",
        yaxis_title=f"Automated − manual{suffix}", template="plotly_white",
    )
    return fig


def difference_distribution_figure(
    frame: pd.DataFrame, config: AgreementConfig = AgreementConfig(),
) -> Any:
    """Plot paired differences and a normal Q-Q diagnostic without assuming normality."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data: pd.DataFrame = _validate_pairs(frame, config)
    reference: np.ndarray = data[config.reference_col].to_numpy(dtype=float)
    prediction: np.ndarray = data[config.prediction_col].to_numpy(dtype=float)
    differences: np.ndarray = prediction - reference
    theoretical, ordered = probplot(differences, dist='norm', fit=False)
    histogram = go.Histogram(x=differences, name='Paired differences')
    quantiles = go.Scatter(x=theoretical, y=ordered, mode='markers', name='Normal Q-Q')
    figure = make_subplots(rows=1, cols=2, subplot_titles=('Difference distribution', 'Normal Q-Q'))
    figure.add_trace(histogram, row=1, col=1)
    figure.add_trace(quantiles, row=1, col=2)
    figure.update_xaxes(title_text=f'Prediction - reference ({config.unit})', row=1, col=1)
    figure.update_xaxes(title_text='Theoretical normal quantile', row=1, col=2)
    figure.update_yaxes(title_text=f'Observed difference ({config.unit})', row=1, col=2)
    return figure


@dataclass(frozen=True)
class ClassificationResult:
    """Binary subject-level counts and exact Clopper-Pearson intervals.

    Confusion rows are truth [0, 1], columns prediction [0, 1]. Every input row
    must represent an independent animal; repeated scans/folds are rejected.
    Undefined rates and intervals are NaN/None, never a perfect score.
    """

    n: int
    confusion_matrix: list[list[int]]
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    ci: dict[str, tuple[float, float] | None]


def calculate_classification(
    frame: pd.DataFrame, *, subject_col: str = 'Subject',
    reference_col: str = 'Reference', prediction_col: str = 'Prediction',
    confidence: float = 0.95,
) -> ClassificationResult:
    """Score explicit binary decisions for independent subjects, not probabilities.

    Apply a prespecified threshold/aggregation before calling. Confidence
    intervals quantify binomial uncertainty, including for 100% accuracy.
    Select one intended cohort/model/decision per subject before calling.
    """
    columns: list[str] = [subject_col, reference_col, prediction_col]
    if len(set(columns)) != 3 or not set(columns).issubset(frame.columns):
        raise ValueError('Provide distinct subject, reference and prediction columns')
    data: pd.DataFrame = frame.loc[:, columns]
    if data.empty or data.isna().any().any():
        raise ValueError('Classification records must be nonempty without missing values')
    if data[subject_col].duplicated().any():
        raise ValueError('Duplicate subjects: aggregate repeated scans/folds explicitly first')
    if not 0 < confidence < 1:
        raise ValueError('confidence must lie strictly between zero and one')

    reference: np.ndarray = data[reference_col].to_numpy()
    prediction: np.ndarray = data[prediction_col].to_numpy()
    if not np.isin(reference, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError('Classification decisions must be binary zero/one labels')

    tn: int = int(np.count_nonzero((reference == 0) & (prediction == 0)))
    fp: int = int(np.count_nonzero((reference == 0) & (prediction == 1)))
    fn: int = int(np.count_nonzero((reference == 1) & (prediction == 0)))
    tp: int = int(np.count_nonzero((reference == 1) & (prediction == 1)))
    counts: dict[str, tuple[int, int]] = {
        'accuracy': (tn + tp, len(data)), 'sensitivity': (tp, tp + fn),
        'specificity': (tn, tn + fp), 'precision': (tp, tp + fp),
    }
    rates: dict[str, float] = {}
    intervals: dict[str, tuple[float, float] | None] = {}
    for metric, (successes, total) in counts.items():
        rates[metric] = float('nan')
        intervals[metric] = None
        if total:
            rates[metric] = successes / total
            binomial = binomtest(successes, total)
            interval = binomial.proportion_ci(confidence_level=confidence, method='exact')
            intervals[metric] = (float(interval.low), float(interval.high))

    confusion: list[list[int]] = [[tn, fp], [fn, tp]]
    return ClassificationResult(n=len(data), confusion_matrix=confusion, ci=intervals, **rates)


def _json_finite(value: Any) -> Any:
    """Represent undefined statistical quantities as standard JSON nulls."""
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def export_agreement(frame: pd.DataFrame, output_dir: Path, config: AgreementConfig = AgreementConfig()) -> Path:
    """Write summary CSV/JSON and per-cohort Plotly HTML where applicable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _validate_pairs(frame, config)
    summary = agreement_table(data, config)
    summary.to_csv(output_dir / "agreement_summary.csv", index=False)
    (output_dir / "analysis_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    if not config.group_cols:
        result = calculate_agreement(data, config)
        payload = _json_finite(asdict(result))
        (output_dir / "agreement_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        try:
            bland_altman_figure(data, result, config).write_html(output_dir / "bland_altman.html")
            distribution = difference_distribution_figure(data, config)
            distribution.write_html(output_dir / 'difference_distribution.html')
        except ImportError:
            pass
    else:
        for idx, (_, subset) in enumerate(data.groupby(list(config.group_cols), dropna=False)):
            result = calculate_agreement(subset, config)
            try:
                bland_altman_figure(subset, result, config).write_html(output_dir / f"bland_altman_group_{idx}.html")
                distribution = difference_distribution_figure(subset, config)
                distribution.write_html(output_dir / f'difference_distribution_group_{idx}.html')
            except ImportError:
                break
    return output_dir


def main() -> None:
    """Command-line entry point for CSV measurement agreement reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV containing paired measurements")
    parser.add_argument("--output", type=Path, default=Path("agreement_report"))
    parser.add_argument("--subject", default="Subject")
    parser.add_argument("--reference", default="Manual")
    parser.add_argument("--prediction", default="Automated")
    parser.add_argument("--group", nargs="*", default=[])
    parser.add_argument("--unit", default="")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paired-t", action="store_true", help="Also report the optional paired t-test")
    args = parser.parse_args()
    config = AgreementConfig(
        subject_col=args.subject, reference_col=args.reference,
        prediction_col=args.prediction, group_cols=tuple(args.group),
        unit=args.unit, bootstrap_samples=args.bootstrap, seed=args.seed,
        include_paired_t=args.paired_t,
    )
    output = export_agreement(pd.read_csv(args.csv), args.output, config)
    print(f"Agreement reports saved to {output.resolve()}")


if __name__ == "__main__":
    main()
