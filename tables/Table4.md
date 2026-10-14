# Table 4 candidate: upstream feature-ranking sensitivity

| Rank | LightGBM | XGBoost | RandomForest |
| --- | --- | --- | --- |
| 1 | SMOKING (0.399492) | ENERGY_LEVEL (0.210566) | SMOKING (0.366240) |
| 2 | THROAT_DISCOMFORT (0.364726) | BREATHING_ISSUE (0.199724) | THROAT_DISCOMFORT (0.233721) |
| 3 | BREATHING_ISSUE (0.254375) | SMOKING (0.186664) | BREATHING_ISSUE (0.203266) |
| 4 | ENERGY_LEVEL (0.227033) | THROAT_DISCOMFORT (0.165600) | SMOKING_FAMILY_HISTORY (0.153918) |
| 5 | STRESS_IMMUNE (0.102125) | EXPOSURE_TO_POLLUTION (0.062624) | STRESS_IMMUNE (0.131469) |
| 6 | FAMILY_HISTORY (0.062675) | STRESS_IMMUNE (0.057360) | ENERGY_LEVEL (0.117171) |
| 7 | EXPOSURE_TO_POLLUTION (0.052212) | FAMILY_HISTORY (0.053034) | EXPOSURE_TO_POLLUTION (0.018303) |

**Table 4. Training-only feature-ranking sensitivity across three upstream models.** All three models used the same 1,000 training records and the same 17 standardized inputs from repeat 1, fold 1. LightGBM reused the frozen median-quantile model; XGBoost used median-quantile regression and Random Forest used an absolute-error split criterion. Mean absolute exact TreeSHAP contributions to standardized-target model outputs are shown in parentheses. Model settings were fixed before this analysis; no held-out record was fitted, explained or scored. This is a representative-fold model-choice analysis. Mean absolute contributions and normalized importance for all 17 inputs, model settings and membership are supplied separately.

## Pairwise agreement

| model_A | model_B | top7_overlap_n | jaccard | spearman | kendall_tau_b |
| --- | --- | --- | --- | --- | --- |
| LightGBM | XGBoost | 7 | 1.0000 | 0.8603 | 0.6922 |
| LightGBM | RandomForest | 6 | 0.7500 | 0.7752 | 0.6742 |
| XGBoost | RandomForest | 6 | 0.7500 | 0.8407 | 0.6324 |

Agreement uses all 17 inputs. Values below 1e-12 are treated as numerical zero for average-tied-rank Spearman/Kendall calculations; exported raw contributions and Top-7 selection retain their original values. Rank coefficients are descriptive.

Suggested result text: “In the representative training fold, LightGBM and XGBoost selected the same seven inputs, while Random Forest shared six of these inputs and selected SMOKING_FAMILY_HISTORY in place of FAMILY_HISTORY. Pairwise Spearman correlations across the 17-input importance rankings ranged from 0.775 to 0.860. These findings indicate agreement in the selected input set under the examined model choices, with differences in the relative ordering and the allocation between a constituent input and its product feature.”
