# SHAP-guided IBRB reproducibility materials

This directory is a local release candidate prepared on 12 October 2026. Repository addresses, permanent identifiers, creator metadata and licensing will be completed before public release.

The original benchmark uses two repetitions of stratified five-fold evaluation. Each outer evaluation fold contains 1,000 records; SHAP and IBRB share a designated 1,000-record subset of the outer training pool. Feature selection and regularization selection are performed within training data. The model uses seven selected inputs, information-gain partitions, continuous soft matching, binary hard matching and 200 CMA generations. The regularization coefficient is selected through three inner folds.

The independent NLST analysis includes 6,996 patients, with 5,200 development patients and 1,796 confirmation patients. SHAP and IBRB each use 1,000 training patients. Support and optimization settings are selected within training data; the regularization coefficient is fixed at 0.05. The NLST model is refitted for its own inputs and outcome.

## Run

Use Python and install `requirements.txt` for frozen-model reconstruction. Optional full training, baseline and acquisition dependencies are described in the companion instructions.

```powershell
python reproduce.py
python train_pipeline.py --help
python train_baselines.py --help
python acquire_data.py --help
python build_nlst_cohort.py --help
```

`reproduce.py` reconstructs original benchmark metrics for 220 model/fold conditions, restores 21 primary models and checks NLST confirmation comparisons. `train_pipeline.py` regenerates upstream SHAP selection and inner tuning; `train_baselines.py` trains comparison models. See `全流程运行说明.md` and `基线训练与数据获取说明.md` for commands and environment details.

Representative complete pipelines and baseline training were verified in the local package. Full training across all folds is supported; the integration check reran representative stages. Validation reports retain this scope.

## Materials

- `data/`, `config/`, `selection/`: exact processed inputs, split membership and training choices.
- `models/`, `predictions/`, `results/`: fitted models, predictions and numerical summaries.
- `figures/`, `tables/`, `source_data/`: candidate graphics, tables, rule inference examples and underlying values.
- `figure_source_mapping.csv`, `table_source_mapping.csv`, `statistical_source_mapping.csv`: relative paths connecting artifacts to sources.
- `docs/`, `provenance/`, `validation/`: metadata, historical provenance and verification records.

The benchmark source match and literature-based interpretation of `PULMONARY_DISEASE` are documented. Original collection or generation procedures and label assignment remain separate provenance questions. The NLST outcome and cohort construction are documented in the acquisition and rebuilding materials. Read `Data_Code_Availability_候选.md` and `仓库与引用安排.md` before preparing the public release.

Historical absolute paths in provenance documents use the marker `historical_workspace`. Executable configuration paths are relative to this package. Figure/table numbering remains provisional until the consolidated manuscript revision.
