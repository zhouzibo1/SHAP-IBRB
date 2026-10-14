# 数据与代码可得性候选说明

本文件供最后统一修稿使用。当前材料保存在本地；归档段落须在实际发布并填写地址后采用。

## Data Availability：来源段落

The original benchmark corresponds to version 1 of the Lung Cancer Prediction Dataset hosted on Kaggle by adnananalytics (https://www.kaggle.com/datasets/adnananalytics/lung-cancer-prediction-dataset). A matching public copy is available in the MatSci-ML-Studio1 repository (https://github.com/wy314159-lyq/MatSci-ML-Studio1/blob/main/case2.csv). Following the interpretation reported by Lam et al. (https://doi.org/10.1002/eng2.70486), we treat PULMONARY_DISEASE as a binary lung cancer target and evaluate classification using the dataset-provided labels (1: positive; 0: negative).

The independent NLST analysis uses publicly accessible clinical tables from Imaging Data Commons, version 21. The acquisition scripts identify the nlst_prsn, nlst_screen, nlst_ctab and nlst_canc tables and their download locations. The processed cohort, patient-level split membership and cohort construction code are included in the reproducibility materials.

## Data Availability：发布后归档段落

The processed analysis inputs, evaluation split membership, model predictions, statistical summaries and numerical values underlying the figures and tables are available at {{ARCHIVE_RECORD_URL}} (version-specific DOI: {{VERSION_DOI}}). Dataset metadata and acquisition instructions are provided with the archived materials.

## Code Availability：发布后归档段落

The analysis code, frozen configurations, fitted models and environment specifications are available at {{CODE_REPOSITORY_URL}}. The version used for this study is archived at {{ARCHIVE_RECORD_URL}} ({{VERSION_DOI}}). The archive includes commands for numerical reconstruction, training with inner selection, baseline comparisons and NLST cohort rebuilding.

## 中文核对

数据来源匹配和标签的文献解释已有记录。Lam等人的表述支持本文采用的目标解释；其作用不延伸至原始采集流程或临床标签判定过程。源数据原始生成/采集及赋值说明仍以来源材料实际提供的信息为准。

归档材料已含精确分析输入。公开前需落实原始数据及衍生材料的分发条款，确认作者、许可及适用的伦理说明。取得真实仓库地址和版本DOI后替换占位符，并在投稿系统的Data Availability字段与正文中保持一致。

PLOS ONE要求提供支撑结果的最小数据集，包含图表底层数值与必要方法、元数据，推荐使用数据仓库：[PLOS ONE Data Availability](https://journals.plos.org/plosone/s/data-availability)。软件说明按期刊[代码共享政策](https://journals.plos.org/plosone/s/materials-software-and-code-sharing)核对。上述英文段落为候选稿，原论文尚未修改。
