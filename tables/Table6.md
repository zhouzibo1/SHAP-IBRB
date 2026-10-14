# Table 6 candidate: original benchmark comparison

| Method | Accuracy (%) | ROC AUC | Brier | Sensitivity (%) | F1 |
| --- | --- | --- | --- | --- | --- |
| Proposed SHAP–IBRB | 88.15 ± 1.86 | 0.91490 ± 0.01033 | 0.10173 ± 0.00857 | 85.13 ± 2.57 | 0.85413 ± 0.02254 |
| BRB | 87.92 ± 2.05 | 0.91119 ± 0.01321 | 0.09991 ± 0.01329 | 85.54 ± 3.31 | 0.85219 ± 0.02565 |
| R-NBRB | 87.07 ± 2.03 | 0.90791 ± 0.01596 | 0.10543 ± 0.01373 | 84.12 ± 3.38 | 0.84118 ± 0.02553 |
| Equal-width IBRB | 86.14 ± 1.30 | 0.90697 ± 0.01187 | 0.11089 ± 0.00737 | 83.48 ± 2.58 | 0.83065 ± 0.01638 |
| XGBoost | 88.11 ± 2.06 | 0.91255 ± 0.01433 | 0.09858 ± 0.01297 | 84.93 ± 3.48 | 0.85322 ± 0.02669 |
| LightGBM | 86.77 ± 2.11 | 0.90687 ± 0.01395 | 0.10685 ± 0.01216 | 80.39 ± 4.31 | 0.83157 ± 0.02919 |
| EBM | 88.20 ± 1.52 | 0.91767 ± 0.00964 | 0.09907 ± 0.00825 | 84.93 ± 2.48 | 0.85436 ± 0.01826 |
| DNN | 88.70 ± 2.32 | 0.91739 ± 0.01375 | 0.09246 ± 0.01482 | 85.10 ± 3.54 | 0.85980 ± 0.02907 |
| CNN | 88.55 ± 2.17 | 0.91491 ± 0.01161 | 0.09415 ± 0.01407 | 85.05 ± 3.25 | 0.85817 ± 0.02684 |
| Logistic regression | 88.35 ± 1.62 | 0.91780 ± 0.00897 | 0.10207 ± 0.00664 | 86.60 ± 1.74 | 0.85845 ± 0.01799 |
| SVM | 86.37 ± 2.44 | 0.90999 ± 0.02281 | 0.11403 ± 0.01714 | 82.21 ± 4.76 | 0.83044 ± 0.03305 |
| Decision tree | 84.94 ± 2.09 | 0.87906 ± 0.00990 | 0.12231 ± 0.00978 | 82.50 ± 3.12 | 0.81706 ± 0.02384 |

Mean ± sample SD over ten outer folds from two repeated stratified five-fold evaluations. Each model uses the same 1,000 designated training records and seven fold-specific selected inputs; evaluation is on 1,000 records per fold. All 22 conditions, model settings, calibration and paired intervals/corrections belong to supplementary tables. SD describes fold variation, not an independent-sample standard error. Brier equals MSE of original class-1 probabilities against binary outcomes.
