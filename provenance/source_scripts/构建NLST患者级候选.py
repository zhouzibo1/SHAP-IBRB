"""构建公开NLST一年诊断分类候选；不拟合模型、不改动旧实验。"""
from pathlib import Path
import sys
import json
import zipfile
import io
import re

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '工具依赖'))
import duckdb
import pandas as pd

SOURCE = ROOT / 'NLST公开子集'
OUTPUT = ROOT / 'NLST患者级候选_20261005'
OUTPUT.mkdir(exist_ok=True)

def read_table(name):
    return duckdb.sql('SELECT * FROM read_parquet(?)', params=[str(SOURCE / f'{name}.parquet')]).df()

person = read_table('nlst_prsn')
screen = read_table('nlst_screen')
abnormality = read_table('nlst_ctab')
cancer = read_table('nlst_canc')
baseline = abnormality[(abnormality.study_yr == 0) & (abnormality.sct_ab_desc == 51)].copy()
baseline_first = baseline[baseline.sct_found_after_comp.astype(str).eq('0')].copy()
for column in ['sct_long_dia', 'sct_perp_dia', 'sct_ab_num', 'sct_epi_loc', 'sct_margins', 'sct_pre_att']:
    baseline_first[column] = pd.to_numeric(baseline_first[column], errors='coerce')

# 最大已知长径对应的结节；并列时按异常编号升序。所有长径缺失时保留编号最小的记录。
ordered = baseline_first.sort_values(['pid', 'sct_long_dia', 'sct_ab_num'], ascending=[True, False, True], na_position='last')
index_nodule = ordered.drop_duplicates('pid').copy()
counts = baseline_first.groupby('pid').size().rename('baseline_nodule_count')
index_nodule = index_nodule.merge(counts, on='pid', validate='one_to_one')
index_nodule = index_nodule.rename(columns={
    'sct_long_dia': 'index_long_diameter_mm', 'sct_perp_dia': 'index_perpendicular_diameter_mm',
    'sct_epi_loc': 'index_lobe', 'sct_margins': 'index_margin', 'sct_pre_att': 'index_attenuation',
    'sct_ab_num': 'index_abnormality_number',
})
selected = ['pid', 'baseline_nodule_count', 'index_long_diameter_mm', 'index_perpendicular_diameter_mm',
            'index_lobe', 'index_margin', 'index_attenuation', 'index_abnormality_number']
cohort = person.merge(index_nodule[selected], on='pid', how='inner', validate='one_to_one')
screen0 = screen[screen.study_yr == 0][['pid']]
cohort = cohort.merge(screen0.assign(has_baseline_ct=True), on='pid', how='left', validate='one_to_one')
for column in ['age', 'gender', 'race', 'cigsmok', 'scr_days0', 'candx_days', 'canc_free_days', 'can_scr']:
    cohort[column] = pd.to_numeric(cohort[column], errors='coerce')
cohort['male'] = cohort.gender.map({1: 1, 2: 0})
cohort['race'] = cohort.race.where(cohort.race.isin(range(1, 8)))
cohort['index_lobe'] = cohort.index_lobe.where(cohort.index_lobe.isin([1, 2, 3, 4, 5, 6, 8]))
cohort['index_margin'] = cohort.index_margin.where(cohort.index_margin.isin([1, 2, 3]))
cohort['index_attenuation'] = cohort.index_attenuation.where(cohort.index_attenuation.isin([1, 2, 3, 4, 6, 7]))
cohort['days_from_baseline_to_diagnosis'] = cohort.candx_days - cohort.scr_days0
cohort['one_year_cutoff_day'] = cohort.scr_days0 + 365
cohort['diagnosis_consistent'] = cohort.candx_days.notna().eq(cohort.can_scr.gt(0))
cohort['cancer_table_match'] = cohort.pid.isin(cancer.pid)

cohort['eligibility_status'] = 'insufficient_outcome_observation'
ready = cohort.has_baseline_ct.eq(True) & cohort.scr_days0.notna() & cohort.diagnosis_consistent
delta = cohort.days_from_baseline_to_diagnosis
positive = ready & delta.between(0, 365) & cohort.cancer_table_match
late = ready & delta.gt(365) & cohort.cancer_table_match
observed_free = ready & cohort.can_scr.eq(0) & cohort.candx_days.isna() & cohort.canc_free_days.ge(cohort.one_year_cutoff_day)
cohort.loc[positive, 'eligibility_status'] = 'confirmed_diagnosis_within_365_days'
cohort.loc[late, 'eligibility_status'] = 'first_confirmed_diagnosis_after_365_days'
cohort.loc[observed_free, 'eligibility_status'] = 'no_confirmed_cancer_with_documented_free_status_beyond_cutoff'
cohort.loc[delta.lt(0), 'eligibility_status'] = 'diagnosis_before_baseline'
cohort.loc[~ready, 'eligibility_status'] = 'missing_baseline_or_inconsistent_diagnosis'
cohort['label_365d'] = pd.Series(pd.NA, index=cohort.index, dtype='Int64')
cohort.loc[positive, 'label_365d'] = 1
cohort.loc[late | observed_free, 'label_365d'] = 0

features = ['age', 'male', 'race', 'cigsmok', 'baseline_nodule_count', 'index_long_diameter_mm',
            'index_perpendicular_diameter_mm', 'index_lobe', 'index_margin', 'index_attenuation']
metadata = ['pid', 'index_abnormality_number', 'scr_days0', 'candx_days', 'canc_free_days', 'can_scr',
            'days_from_baseline_to_diagnosis', 'one_year_cutoff_day', 'eligibility_status', 'label_365d']
cohort = cohort.sort_values('pid')
cohort[features + metadata].to_csv(OUTPUT / '全部候选与纳入状态.csv', index=False, encoding='utf-8-sig')
eligible = cohort[cohort.label_365d.notna()]
eligible[['pid'] + features + ['label_365d']].to_csv(OUTPUT / '患者级建模候选.csv', index=False, encoding='utf-8-sig')
profile = {
    'candidate_patients': int(baseline.pid.nunique()),
    'first_read_nodule_records': len(baseline_first),
    'first_read_patients': len(cohort),
    'eligible_patients': len(eligible),
    'positive': int(eligible.label_365d.eq(1).sum()),
    'negative': int(eligible.label_365d.eq(0).sum()),
    'status_counts': cohort.eligibility_status.value_counts().to_dict(),
    'feature_missing_counts': eligible[features].isna().sum().to_dict(),
    'numeric_summary': eligible[features].describe().to_dict(),
    'raw_diameter_below_4mm_records': int(baseline_first.sct_long_dia.lt(4).sum()),
    'same_day_diagnoses': int(delta.eq(0).sum()),
    'diagnosis_date_status_disagreements': int((~cohort.diagnosis_consistent).sum()),
    'diagnosed_persons_absent_from_cancer_table': int((cohort.candx_days.notna() & ~cohort.cancer_table_match).sum()),
    'training_started': False,
}
(OUTPUT / '队列核查结果.json').write_text(json.dumps(profile, ensure_ascii=False, indent=2, default=int), encoding='utf-8')
config = {
    'status': 'candidate_dataset_for_review', 'source_version': 'IDC v21 clinical',
    'time_zero': 'baseline CT (study_yr=0), first-read noncalcified nodule or mass (sct_ab_desc=51)',
    'horizon_days': 365,
    'outcome': 'confirmed lung cancer diagnosis within 365 days after baseline CT',
    'negative_definition': 'first confirmed diagnosis after cutoff, or no confirmed cancer and documented cancer-free status at/after cutoff',
    'insufficient_outcome_observation': 'exclude from this classification cohort',
    'canc_free_days_use': 'documented status eligibility only; no incidence rate, Cox model, or survival follow-up reconstruction',
    'index_nodule_rule': 'largest available baseline first-read long diameter, tie by abnormality number',
    'feature_columns': features,
    'nominal_columns': ['race', 'index_lobe', 'index_margin', 'index_attenuation'],
    'label_column': 'label_365d', 'identifier_column': 'pid',
    'encoding_and_imputation': 'fit later using training data only',
    'desired_training_budgets': {'SHAP': 1000, 'IBRB': 260, 'test': 130},
    'split_created': False,
}
(OUTPUT / '候选队列方案.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
with zipfile.ZipFile(SOURCE / 'TCIA临床包.zip') as outer:
    with zipfile.ZipFile(io.BytesIO(outer.read('nlst_780/nlst780.idc.delivery.052821.zip'))) as inner:
        for name in inner.namelist():
            if name.startswith('dictionary_idc_prsn') or name.startswith('dictionary_idc_ctab'):
                text = inner.read(name).decode('latin1')
                text = re.sub(r'\\[a-zA-Z]+-?\d* ?', ' ', text).replace('{', '').replace('}', '')
                (OUTPUT / (Path(name).stem + '.txt')).write_text(re.sub(r'\s+', ' ', text), encoding='utf-8')
assert eligible.pid.is_unique
assert len(eligible) == int(positive.sum() + late.sum() + observed_free.sum())
print(json.dumps({k: profile[k] for k in ['eligible_patients', 'positive', 'negative', 'status_counts', 'feature_missing_counts']}, ensure_ascii=False))
