# Data and code availability

## Code and computational records

The package includes analysis code, dependency versions, frozen configurations, split membership, fitted models, original predictions, statistical summaries and numerical figure/table sources. Code is licensed under MIT; data and output licensing are described in THIRD_PARTY_NOTICES.md. Repository URL and version-specific archive DOI will be inserted after the corresponding records are created. No public deposit is claimed in this local package.

## Original benchmark

The analyzed 5,000-record benchmark matches version 1 of the Lung Cancer Prediction Dataset hosted by adnananalytics on Kaggle: https://www.kaggle.com/datasets/adnananalytics/lung-cancer-prediction-dataset . A matching public copy is available at https://github.com/wy314159-lyq/MatSci-ML-Studio1/blob/main/case2.csv . Following Lam et al. (https://doi.org/10.1002/eng2.70486), the analysis uses PULMONARY_DISEASE as the literature-interpreted binary lung-cancer target (1 positive, 0 negative). The source does not provide original collection/generation and diagnostic-assignment documentation. This target interpretation is distinct from an independently verified diagnostic reference standard.

data/original.csv preserves the numerical precision of the frozen analysis inputs. Public copies have small continuous-value precision differences (approximately 5e-9 or less); these can affect optimizer trajectories. The original benchmark's redistribution terms remain unresolved. Keep its exact analysis file in the local working package until the public redistribution scope is settled; use the source links and acquisition scripts to document access.

```powershell
python acquire_data.py --dataset original --provider kaggle --output downloaded_original
python acquire_data.py --dataset original --provider matsci --output downloaded_matsci
```

## NLST

NLST public clinical data and dictionaries are distributed by TCIA under CC BY 4.0. The dataset citation is National Lung Screening Trial Research Team (2013), Data from the National Lung Screening Trial (NLST), https://doi.org/10.7937/TCIA.HMQ8-J677 . See THIRD_PARTY_NOTICES.md for the source, license, publications and changes made for this analysis.

This study uses IDC v21 public clinical tables nlst_prsn, nlst_screen, nlst_ctab and nlst_canc. data/nlst.csv contains the processed 6,996-patient baseline cohort. The outcome is first lung cancer diagnosis within 365 days after baseline CT. The development/confirmation split is patient-level (5,200/1,796). The processed NLST clinical subset retains CC BY 4.0 attribution and change notices.

```powershell
python acquire_data.py --dataset nlst --build --output downloaded_nlst
```

## Ethics

The original NLST trial obtained approval at the 33 participating institutions and written informed consent. This study analyzes the publicly available de-identified clinical subset, with no new recruitment, patient contact, intervention or data collection. Under the applicable institutional policy confirmed by the authors, no additional ethics approval was required for this secondary analysis. Source trial references: https://doi.org/10.1056/NEJMoa1102873 and https://doi.org/10.1056/NEJMoa1211776 .

The original benchmark source does not report ethics approval, consent or de-identification procedures. The NLST ethics and licensing statements apply to the NLST analysis.
