# SHAP-guided IBRB reproducibility materials

Code and computational records accompanying *A hybrid interpretable risk assessment method for lung cancer integrating SHAP with a supervised interval belief rule base*.

Authors: Shaohua Li, Tianshu Guan, Zibo Zhou, Dongkun Liu, Jingying Feng, Ruihua Qi, Wei He. See [AUTHORS.md](AUTHORS.md) for affiliations and corresponding-author information.

## Evaluation protocol

The original 5,000-record benchmark uses two repetitions of stratified five-fold evaluation. Each fold has 1,000 evaluation records; SHAP and IBRB share a designated 1,000-record fitting subset of the outer training pool. Preprocessing, SHAP and information-gain partitioning are fitted within training data. Three inner folds select the regularization coefficient from 0, 0.01, 0.05 and 0.20. The model uses seven selected inputs, continuous soft matching, binary hard matching, adjacent-rule smoothing and 200 CMA-ES generations. Mean accuracy is 88.15%, AUC 0.914901 and Brier score 0.101731.

The NLST analysis uses 6,996 baseline-CT patients, with 5,200 development and 1,796 confirmation patients. SHAP and IBRB share 1,000 training patients; support and optimization settings are selected within training data, with regularization fixed at 0.05. Confirmation AUC is 0.853105 and Brier score 0.030397. The outcome and predictors are fitted and defined for NLST separately from the original benchmark.

## Quick start

Use Python 3.12 and run commands from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py
```

On Linux/macOS activate with `source .venv/bin/activate`. Frozen-model reconstruction checks 220 original-benchmark model/fold conditions, restores 10 original and 11 NLST primary models, and checks nine NLST confirmation comparison conditions. Its saved verification record is in `preparation_verification.json`. Full training was checked on representative stages; all 21 nested training procedures were not rerun during integration verification.

Exact original-benchmark model reconstruction requires `data/original.csv`; read [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for its source, precision and redistribution status before preparing the public upload.

## Retraining and acquisition

```powershell
python train_pipeline.py --task both --stage repeat1_fold1
python train_pipeline.py --task both --all --output all_pipeline_output
python -m pip install -r requirements-baselines.txt
python train_baselines.py --task original --stage repeat1_fold1 --inputs all
python -m pip install -r requirements-data.txt
python acquire_data.py --dataset nlst --build --output downloaded_nlst
```

See [full training instructions](全流程运行说明.md), [baseline and acquisition instructions](基线训练与数据获取说明.md), and command-line `--help` for supported tasks. Full nested training takes substantially longer than frozen-model reconstruction.

## Package layout

| Directory or file | Contents |
| --- | --- |
| `src/` and Python entry points | Analysis, reconstruction, retraining and acquisition code |
| `config/`, `selection/` | Exact split membership and training choices |
| `data/` | Processed inputs; source-specific rights are in DATA_AVAILABILITY.md |
| `models/`, `predictions/`, `results/` | Frozen models, original predictions and results |
| `figures/`, `tables/`, `source_data/` | Figures, tables and numerical source values |
| `*_source_mapping.csv` | Relative mappings between artifacts and numerical sources |
| `validation/` | Saved reconstruction and representative training checks |
| `provenance/`, `docs/` | Source and methodological records |

Historical source records may refer to the original development workspace; executable configurations use paths relative to this package. Figure and table labels will be aligned with the consolidated revised manuscript before the paper-linked archive version is finalized.

## Licenses and citation

- Project-authored code: [MIT](LICENSE).
- Author-created figures, explanatory text and aggregate analytical summaries: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- NLST public clinical subset: CC BY 4.0 with the attribution and modification notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Original benchmark: access instructions and unresolved redistribution scope in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

[CITATION.cff](CITATION.cff) contains software authorship. Repository and archive identifiers will be added after creation. The NLST dataset DOI is https://doi.org/10.7937/TCIA.HMQ8-J677 .
