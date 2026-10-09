"""Agreement diagnostics and independent-subject classification uncertainty."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import levene, linregress, ttest_rel, wilcoxon

from src.evaluation.statistics import (
    AgreementConfig, agreement_table, calculate_agreement, calculate_classification,
    export_agreement,
)


def paired_frame(reference, prediction):
    return pd.DataFrame({'Subject': [f's{i}' for i in range(len(reference))],
                         'Manual': reference, 'Automated': prediction})


def test_identical_volumes_preserve_agreement_and_define_zero_difference_policy():
    data = paired_frame(np.arange(1., 17.), np.arange(1., 17.))
    config = AgreementConfig(bootstrap_samples=100)
    result = calculate_agreement(data, config)
    assert result.bias == result.mae == result.rmse == 0
    assert result.loa_lower == result.loa_upper == 0
    assert result.ccc == pytest.approx(1)
    assert result.icc_a1 == pytest.approx(1)
    assert result.diagnostics['wilcoxon_pvalue'] == 1
    assert result.sd_reduction_percent == 0
    assert result.ci['sd_reduction_percent'] == (0., 0.)


def test_paired_bootstrap_preserves_scaling_and_reproducibility():
    reference = np.arange(1., 17.)
    prediction = 0.81 * reference + 2
    data = paired_frame(reference, prediction)
    config = AgreementConfig(bootstrap_samples=100, seed=7)
    result = calculate_agreement(data, config)
    repeated = calculate_agreement(data, config)
    assert result.ci == repeated.ci
    assert result.sd_reduction_percent == pytest.approx(19)
    np.testing.assert_allclose(result.ci['sd_reduction_percent'], (19, 19), atol=1e-12)
    assert result.proportional_slope == pytest.approx(2 * (0.81 - 1) / (0.81 + 1))
    assert result.pearson_r == pytest.approx(1)
    assert 'independent' in result.diagnostics['levene_assumption']


def test_tests_match_scipy_on_nondegenerate_pairs():
    reference = np.array([1., 3., 6., 8., 13., 17., 22., 26.])
    prediction = np.array([2., 5., 5., 11., 9., 18., 27., 24.])
    data = paired_frame(reference, prediction)
    config = AgreementConfig(bootstrap_samples=100, include_paired_t=True)
    result = calculate_agreement(data, config)
    difference = prediction - reference
    expected_w = wilcoxon(difference, method='auto')
    expected_l = levene(reference, prediction, center='median')
    expected_t = ttest_rel(prediction, reference)
    expected_slope = linregress((prediction + reference) / 2, difference)
    assert result.diagnostics['wilcoxon_pvalue'] == expected_w.pvalue
    assert result.diagnostics['levene_pvalue'] == expected_l.pvalue
    assert result.diagnostics['paired_t_pvalue'] == expected_t.pvalue
    assert result.proportional_slope == pytest.approx(expected_slope.slope)
    assert result.rmse == pytest.approx(np.sqrt(np.mean(difference ** 2)))


def test_constant_reference_and_json_undefined_policy(tmp_path):
    data = paired_frame([2., 2., 2., 2.], [2., 2., 2., 2.])
    config = AgreementConfig(bootstrap_samples=100)
    result = calculate_agreement(data, config)
    assert np.isnan(result.sd_reduction_percent)
    assert result.ci['sd_reduction_percent'] is None
    output = export_agreement(data, tmp_path, config)
    raw = (output / 'agreement_results.json').read_text()
    assert 'NaN' not in raw and 'Infinity' not in raw
    payload = json.loads(raw)
    assert payload['sd_reduction_percent'] is None
    summary = pd.read_csv(output / 'agreement_summary.csv')
    assert 'wilcoxon_pvalue' in summary
    assert 'proportional_slope_ci_lower' in summary


def test_duplicate_animals_rejected_by_agreement_and_classification():
    data = paired_frame([1., 2., 3.], [1., 2., 4.])
    data.loc[1, 'Subject'] = data.loc[0, 'Subject']
    with pytest.raises(ValueError, match='Duplicate'):
        calculate_agreement(data)
    with pytest.raises(ValueError, match='Duplicate'):
        calculate_classification(data, reference_col='Manual', prediction_col='Automated')


def test_perfect_classification_still_has_uncertainty_with_16_animals():
    labels = [0] * 8 + [1] * 8
    frame = pd.DataFrame({'Subject': range(16), 'Reference': labels, 'Prediction': labels})
    result = calculate_classification(frame)
    assert result.confusion_matrix == [[8, 0], [0, 8]]
    assert result.accuracy == result.sensitivity == result.specificity == 1
    assert result.ci['accuracy'][0] == pytest.approx(0.025 ** (1 / 16))
    assert result.ci['accuracy'][1] == 1
    assert result.ci['sensitivity'][0] < result.ci['accuracy'][0]


def test_classification_zero_denominators_and_confusion_orientation():
    frame = pd.DataFrame({'Subject': range(3), 'Reference': [0, 0, 1], 'Prediction': [0, 0, 0]})
    result = calculate_classification(frame)
    assert result.confusion_matrix == [[2, 0], [1, 0]]
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.sensitivity == 0
    assert np.isnan(result.precision)
    assert result.ci['precision'] is None
    assert result.ci['sensitivity'][1] > 0


def test_distribution_figure_and_grouped_reports(tmp_path):
    from src.evaluation.statistics import difference_distribution_figure
    data = paired_frame(np.arange(1., 9.), np.arange(1., 9.) * 0.9)
    data['Dataset'] = ['D1'] * 4 + ['D2'] * 4
    config = AgreementConfig(bootstrap_samples=100, group_cols=('Dataset',))
    figure = difference_distribution_figure(data, config)
    assert len(figure.data) == 2
    summary = agreement_table(data, config)
    assert summary['n'].tolist() == [4, 4]
    assert set(summary['Dataset']) == {'D1', 'D2'}
    output = export_agreement(data, tmp_path, config)
    assert (output / 'difference_distribution_group_0.html').exists()
    assert (output / 'bland_altman_group_1.html').exists()
