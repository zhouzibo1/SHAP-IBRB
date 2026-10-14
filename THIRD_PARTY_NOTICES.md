# Licensing and source attribution

## Project software

The project-authored Python code and its software documentation are licensed under MIT; see LICENSE. Python dependencies retain their own licenses. Dependencies are installed separately through the requirements files.

## Author-created research outputs

The authors license their original figures, table designs, explanatory text and independently created aggregate analytical summaries under Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/ . Credit the manuscript authors listed in AUTHORS.md and identify modifications. This grant covers the authors' original contributions; source data and third-party content retain the terms described below.

## NLST public clinical data

TCIA lists the public NLST clinical data, including dictionaries, under CC BY 4.0. The source collection is https://wiki.cancerimagingarchive.net/display/NLST and the license is https://creativecommons.org/licenses/by/4.0/ .

Required dataset attribution: National Lung Screening Trial Research Team. (2013). Data from the National Lung Screening Trial (NLST) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.HMQ8-J677 .

Related trial publication: National Lung Screening Trial Research Team. (2011). Reduced Lung-Cancer Mortality with Low-Dose Computed Tomographic Screening. New England Journal of Medicine, 365, 395–409. https://doi.org/10.1056/NEJMoa1102873 .

TCIA publication: Clark et al. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26, 1045–1057. https://doi.org/10.1007/s10278-013-9622-7 .

Acquisition: IDC version 21 public clinical tables nlst_prsn, nlst_screen, nlst_ctab and nlst_canc. Rebuilding code: acquire_data.py and build_nlst_cohort.py. Changes for this study include joining the clinical tables, selecting the eligible baseline-CT nodule cohort, deriving predictors and the first-lung-cancer-diagnosis-within-365-days outcome, and defining patient-level development and confirmation membership. The resulting data/nlst.csv contains 6,996 patients (5,200 development; 1,796 confirmation). Retain the source attribution, license link and this description of changes when redistributing this processed clinical subset.

This license statement concerns the public clinical subset used here. The full CDAS clinical dataset follows its own access arrangements. TCIA data usage policy: https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/ .

## Original benchmark

Matched public source: https://www.kaggle.com/datasets/adnananalytics/lung-cancer-prediction-dataset (version 1). Matching literature-associated copy: https://github.com/wy314159-lyq/MatSci-ML-Studio1/blob/main/case2.csv .

An explicit redistribution license has not been established for this benchmark. The authors' MIT and CC BY grants do not grant redistribution rights to data/original.csv or third-party input records reproduced elsewhere in the package. Resolve redistribution scope before publicly uploading those records. Acquisition and source-comparison commands are provided in DATA_AVAILABILITY.md and the acquisition guide.
