# Table 9 candidate: record_binary

| Perturbation level | Method | Accuracy (%) | ROC AUC | Brier |
| --- | --- | --- | --- | --- |
| 0 | Equal-width IBRB | 86.14 ± 1.30 | 0.90697 ± 0.01187 | 0.11089 ± 0.00737 |
| 0 | Proposed SHAP–IBRB | 88.15 ± 1.86 | 0.91490 ± 0.01033 | 0.10173 ± 0.00857 |
| 0.05 | Equal-width IBRB | 85.64 ± 1.27 | 0.90240 ± 0.01169 | 0.11417 ± 0.00722 |
| 0.05 | Proposed SHAP–IBRB | 87.61 ± 1.79 | 0.91020 ± 0.01022 | 0.10546 ± 0.00822 |
| 0.1 | Equal-width IBRB | 85.14 ± 1.26 | 0.89772 ± 0.01136 | 0.11756 ± 0.00713 |
| 0.1 | Proposed SHAP–IBRB | 87.06 ± 1.78 | 0.90542 ± 0.00994 | 0.10928 ± 0.00805 |
| 0.15 | Equal-width IBRB | 84.67 ± 1.26 | 0.89294 ± 0.01137 | 0.12088 ± 0.00719 |
| 0.15 | Proposed SHAP–IBRB | 86.53 ± 1.73 | 0.90049 ± 0.00988 | 0.11305 ± 0.00784 |
| 0.2 | Equal-width IBRB | 84.22 ± 1.21 | 0.88860 ± 0.01130 | 0.12395 ± 0.00710 |
| 0.2 | Proposed SHAP–IBRB | 86.02 ± 1.65 | 0.89608 ± 0.00971 | 0.11650 ± 0.00764 |
| 0.25 | Equal-width IBRB | 83.77 ± 1.16 | 0.88429 ± 0.01102 | 0.12703 ± 0.00683 |
| 0.25 | Proposed SHAP–IBRB | 85.53 ± 1.57 | 0.89172 ± 0.00934 | 0.11992 ± 0.00723 |
| 0.3 | Equal-width IBRB | 83.32 ± 1.12 | 0.87966 ± 0.01091 | 0.13032 ± 0.00678 |
| 0.3 | Proposed SHAP–IBRB | 85.02 ± 1.51 | 0.88702 ± 0.00929 | 0.12362 ± 0.00707 |

Ten paired perturbation realizations per fold and level, averaged within each fold before the ten-fold mean ± SD. Continuous level is Gaussian SD divided by the corresponding training-variable SD. Record-binary level is the fraction of records with exactly one primitive binary input flipped; product terms are recomputed. Outcome labels remain unchanged. All seven levels and complete baseline conditions are retained in source data. Fixed LR/LightGBM control settings follow this sensitivity experiment.
