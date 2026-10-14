# Table 7 candidate: NLST development

| Method | Accuracy (%) | ROC AUC | Brier | AP | Sensitivity (%) |
| --- | --- | --- | --- | --- | --- |
| Proposed SHAP–IBRB | 96.04 ± 0.33 | 0.86910 ± 0.04212 | 0.03103 ± 0.00220 | 0.36782 ± 0.05619 | 30.08 ± 6.20 |
| LightGBM | 96.16 ± 0.47 | 0.89050 ± 0.02838 | 0.03136 ± 0.00274 | 0.35229 ± 0.06488 | 27.27 ± 8.06 |
| Logistic regression | 96.40 ± 0.27 | 0.88999 ± 0.02467 | 0.02952 ± 0.00234 | 0.37623 ± 0.05995 | 20.91 ± 7.78 |
| Equal-width IBRB | 96.02 ± 0.36 | 0.82789 ± 0.05841 | 0.03288 ± 0.00329 | 0.31240 ± 0.07383 | 23.97 ± 7.62 |
| Training-prevalence predictor | 96.23 ± 0.04 | 0.50000 ± 0.00000 | 0.03627 ± 0.00036 | 0.03769 ± 0.00039 | 0.00 ± 0.00 |

Development cohort: 5,200 patients; two repeats of patient-level five-fold evaluation; 1,000 designated training patients per fold. The seven encoded terms are selected within training. These development summaries and the independent confirmation results are separate evidence.
