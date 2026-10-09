Almost, but not completely. The two modules cover most of the reviewers' requested statistical analyses, but a few specific tests and segmentation metrics are still missing.

I would keep the existing implementation and add a small number of targeted features rather than expand it into a general statistical library.

## 1. Agreement statistics — `evaluation/statistics.py`

The reviewers primarily want to establish whether automated lesion-volume measurements agree with manual measurements, not merely whether they correlate.

| Statistic                | Status                | Recommendation                      |
| ------------------------ | --------------------- | ----------------------------------- |
| Bland–Altman plot        | &#x20;Implemented     | Keep                                |
| Mean bias                | &#x20;Implemented     | Keep                                |
| 95% limits of agreement  | &#x20;Implemented     | Keep                                |
| MAE                      | &#x20;Implemented     | Keep                                |
| ICC(A,1)                 | &#x20;Implemented     | Keep                                |
| Lin's CCC                | &#x20;Implemented     | Keep                                |
| Bootstrap 95% CIs        | &#x20;Implemented     | Keep                                |
| Paired significance test | &#x20;Missing         | Add                                 |
| Levene's variance test   | &#x20;Missing         | Add                                 |
| Pearson correlation      | &#x20;Not implemented | Optional, already reported in paper |
| RMSE                     | &#x20;Not implemented | Optional                            |

The revision workspace explicitly asks for Bland–Altman, MAE, ICC/CCC, paired analyses and testing of the reported 19% reduction in variability.&#x20;

Major_Revision_BSPC_structured.docx



### Two additions I consider necessary

A. Paired volume comparison

For the same 16 animals measured manually and automatically, add:

- Wilcoxon signed-rank test for paired differences.
- Optionally, a paired t-test when its assumptions are appropriate.

These complement the agreement statistics, although they do not replace Bland–Altman. A nonsignificant paired test does not establish agreement.

B. Variance comparison

Reviewer 1 specifically requests Levene's test for the variability-reduction claim.

Add Levene's test, ideally alongside a robust paired-data sensitivity analysis, because ordinary Levene assumes independent samples whereas the manual and automated measurements come from the same animals.

The module should distinguish the descriptive percentage reduction in SD from statistical evidence that variability genuinely differs.

## 2. Segmentation metrics — K-Fold evaluation and comparison

The current `compare_experiments.py` supports generic numeric metrics, paired comparisons, bootstrap intervals and Holm correction. However, it does not itself calculate segmentation metrics from masks.

The reviewers also want quantitative evaluation of lesions, ventricles and other structures, particularly in full volumes rather than only patches.&#x20;

Major_Revision_BSPC_structured.docx



I would use the following metric set:

| Metric                                    | Importance  | Why                               |
| ----------------------------------------- | ----------- | --------------------------------- |
| Dice coefficient (DSC)                    | Essential   | Primary segmentation overlap      |
| HD95                                      | High        | Robust boundary discrepancy       |
| Average symmetric surface distance (ASSD) | High        | Average anatomical boundary error |
| Absolute volume error (AVE)               | High        | Accuracy of volume estimation     |
| Relative volume error (RVE)               | Recommended | Error relative to structure size  |
| Sensitivity / recall                      | Recommended | Missed lesion voxels              |
| Precision                                 | Recommended | False-positive lesion voxels      |
| IoU                                       | Optional    | Largely redundant with Dice       |

For the reviewers, Dice + HD95 + ASSD + volume error would be my preferred core set.

The metrics should be reported per subject, per anatomical structure and per dataset/model variant. For HD95 and ASSD, empty structures need an explicit, consistent policy; arbitrarily replacing undefined distances with zero would be misleading.

For classification, the reviewers also request confusion matrix, sensitivity, specificity and uncertainty around the reported 100% accuracy.&#x20;

Major_Revision_BSPC_structured.docx

&#x20;These belong in a separate classification-evaluation helper, with confidence intervals calculated at the appropriate independent-subject level.



## 3. Are there unnecessary statistics?

I would not remove the existing statistics, but some are redundant.

ICC and CCC: The reviewers ask for ICC or CCC, so strictly speaking one would satisfy that request. Reporting both is reasonable because their definitions and interpretation differ, but it is not necessary to emphasize both equally.

Pearson correlation: It is already in the manuscript. Retain it for continuity, but do not use it as the principal agreement statistic.

IoU and Dice: For binary segmentation, IoU is a monotonic transformation of Dice. Reporting both adds little scientific information.

RMSE and MAE: MAE is sufficient for the primary volume-error analysis. RMSE is useful only if large individual errors deserve additional emphasis.

## 4. Suggested final module organization

```
evaluation/
├── __init__.py
├── statistics.py
│   ├── Bland–Altman
│   ├── ICC / CCC
│   ├── MAE
│   ├── paired tests
│   └── variance comparison
│
├── segmentation.py
│   ├── Dice / IoU
│   ├── HD95 / ASSD
│   ├── volume errors
│   └── sensitivity / precision
│
└── classification.py
    ├── confusion matrix
    ├── sensitivity / specificity
    └── confidence intervals
```

`compare_experiments.py` should remain separate and consume the standardized outputs of these modules.

## 5. One important statistical caution

Your D4 dataset contains only 16 independent animals. This makes confidence intervals particularly important.

For Bland–Altman, your module currently calculates normal-theory limits of agreement and bootstrap confidence intervals. That is a defensible starting point, but I would add a check for proportional bias (whether measurement differences increase with lesion size), examine the distribution of differences, and consider small-sample uncertainty methods for the limits of agreement.

I would also clearly separate agreement between methods from variance differences between measurement distributions. These are different scientific questions.

## My recommended changes

Choose the additions to implement

Add Wilcoxon paired test

For manual versus automated lesion volumes.

Add Levene and paired-variance sensitivity analysis

For the reported 19% variability reduction.

Add Dice, HD95, ASSD and volume errors

For full-volume anatomical segmentation.

Add classifier confusion matrix and confidence intervals

For the auxiliary classification task.

Add proportional-bias diagnostics

For stronger Bland–Altman interpretation.

Add Pearson and RMSE

Optional descriptive continuity; not core requirements.

Implement selected additions&#x20;

Bottom line: The agreement module already covers the principal requested agreement measures. The remaining statistical gaps are the paired tests and defensible variance analysis. The larger missing functionality is a standardized segmentation-metrics calculator to complement your existing K-Fold comparison pipeline.