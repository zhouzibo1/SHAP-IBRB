# Table 8 candidate: downstream training budget

| Downstream training n | Method | Accuracy (%) | ROC AUC | Brier | Rules |
| --- | --- | --- | --- | --- | --- |
| 100 | Equal-width IBRB | 82.39 ± 2.83 | 0.86674 ± 0.03168 | 0.13636 ± 0.01984 | 140.00000 ± 0.00000 |
| 100 | Proposed SHAP–IBRB | 83.64 ± 3.24 | 0.88959 ± 0.02360 | 0.12681 ± 0.01932 | 33.80000 ± 5.69210 |
| 200 | Equal-width IBRB | 85.21 ± 1.81 | 0.89126 ± 0.01754 | 0.12044 ± 0.01181 | 140.00000 ± 0.00000 |
| 200 | Proposed SHAP–IBRB | 86.55 ± 1.97 | 0.90857 ± 0.01548 | 0.10815 ± 0.01270 | 33.80000 ± 5.69210 |
| 260 | Equal-width IBRB | 85.42 ± 1.41 | 0.89671 ± 0.01550 | 0.11691 ± 0.01039 | 140.00000 ± 0.00000 |
| 260 | Proposed SHAP–IBRB | 86.81 ± 1.44 | 0.90996 ± 0.01274 | 0.10616 ± 0.00981 | 33.80000 ± 5.69210 |
| 300 | Equal-width IBRB | 86.11 ± 1.88 | 0.89624 ± 0.01376 | 0.11574 ± 0.01062 | 140.00000 ± 0.00000 |
| 300 | Proposed SHAP–IBRB | 86.77 ± 1.69 | 0.91112 ± 0.01106 | 0.10583 ± 0.00888 | 33.80000 ± 5.69210 |
| 500 | Equal-width IBRB | 85.96 ± 1.66 | 0.90321 ± 0.01469 | 0.11288 ± 0.00896 | 140.00000 ± 0.00000 |
| 500 | Proposed SHAP–IBRB | 87.76 ± 1.67 | 0.91387 ± 0.01144 | 0.10292 ± 0.00939 | 33.80000 ± 5.69210 |
| 1000 | Equal-width IBRB | 86.14 ± 1.30 | 0.90697 ± 0.01187 | 0.11089 ± 0.00737 | 140.00000 ± 0.00000 |
| 1000 | Proposed SHAP–IBRB | 88.15 ± 1.86 | 0.91490 ± 0.01033 | 0.10173 ± 0.00857 | 33.80000 ± 5.69210 |
| 2000 | Equal-width IBRB | 86.61 ± 1.26 | 0.90825 ± 0.01204 | 0.10926 ± 0.00728 | 140.00000 ± 0.00000 |
| 2000 | Proposed SHAP–IBRB | 87.99 ± 1.72 | 0.91624 ± 0.01058 | 0.10050 ± 0.00916 | 33.80000 ± 5.69210 |
| 3000 | Equal-width IBRB | 86.34 ± 1.23 | 0.90832 ± 0.01201 | 0.10918 ± 0.00780 | 140.00000 ± 0.00000 |
| 3000 | Proposed SHAP–IBRB | 87.70 ± 1.56 | 0.91559 ± 0.01003 | 0.10094 ± 0.00840 | 33.80000 ± 5.69210 |
| 4000 | Equal-width IBRB | 86.94 ± 1.54 | 0.91039 ± 0.00931 | 0.10722 ± 0.00639 | 140.00000 ± 0.00000 |
| 4000 | Proposed SHAP–IBRB | 87.45 ± 1.17 | 0.91574 ± 0.00942 | 0.10105 ± 0.00713 | 33.80000 ± 5.69210 |

Nine nested downstream budgets: 100, 200, 260, 300, 500, 1,000, 2,000, 3,000 and 4,000. Upstream SHAP remains fitted with 1,000 labels and lambda remains the prior per-fold choice. Ten related outer folds; mean ± SD. This assesses downstream fitting budget. Complete six-model results retain the fixed LR/LightGBM controls of this experiment, which differ from the final unified Table 6 selection protocol.
