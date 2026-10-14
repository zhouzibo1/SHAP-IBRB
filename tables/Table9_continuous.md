# Table 9 candidate: continuous

| Perturbation level | Method | Accuracy (%) | ROC AUC | Brier |
| --- | --- | --- | --- | --- |
| 0 | Equal-width IBRB | 86.14 ± 1.30 | 0.90697 ± 0.01187 | 0.11089 ± 0.00737 |
| 0 | Proposed SHAP–IBRB | 88.15 ± 1.86 | 0.91490 ± 0.01033 | 0.10173 ± 0.00857 |
| 0.05 | Equal-width IBRB | 86.15 ± 1.27 | 0.90684 ± 0.01203 | 0.11095 ± 0.00750 |
| 0.05 | Proposed SHAP–IBRB | 88.17 ± 1.79 | 0.91487 ± 0.01041 | 0.10171 ± 0.00849 |
| 0.1 | Equal-width IBRB | 86.19 ± 1.25 | 0.90683 ± 0.01192 | 0.11087 ± 0.00750 |
| 0.1 | Proposed SHAP–IBRB | 88.13 ± 1.78 | 0.91471 ± 0.01066 | 0.10184 ± 0.00861 |
| 0.15 | Equal-width IBRB | 86.19 ± 1.24 | 0.90650 ± 0.01192 | 0.11102 ± 0.00752 |
| 0.15 | Proposed SHAP–IBRB | 88.09 ± 1.74 | 0.91457 ± 0.01080 | 0.10199 ± 0.00869 |
| 0.2 | Equal-width IBRB | 86.19 ± 1.25 | 0.90616 ± 0.01230 | 0.11114 ± 0.00766 |
| 0.2 | Proposed SHAP–IBRB | 88.03 ± 1.70 | 0.91438 ± 0.01088 | 0.10214 ± 0.00875 |
| 0.25 | Equal-width IBRB | 86.14 ± 1.18 | 0.90548 ± 0.01237 | 0.11147 ± 0.00763 |
| 0.25 | Proposed SHAP–IBRB | 88.00 ± 1.71 | 0.91421 ± 0.01107 | 0.10228 ± 0.00886 |
| 0.3 | Equal-width IBRB | 86.05 ± 1.11 | 0.90477 ± 0.01247 | 0.11182 ± 0.00748 |
| 0.3 | Proposed SHAP–IBRB | 87.90 ± 1.71 | 0.91388 ± 0.01120 | 0.10259 ± 0.00891 |

Ten paired perturbation realizations per fold and level, averaged within each fold before the ten-fold mean ± SD. Continuous level is Gaussian SD divided by the corresponding training-variable SD. Record-binary level is the fraction of records with exactly one primitive binary input flipped; product terms are recomputed. Outcome labels remain unchanged. All seven levels and complete baseline conditions are retained in source data. Fixed LR/LightGBM control settings follow this sensitivity experiment.
