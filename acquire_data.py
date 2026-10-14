"""Acquire publicly accessible copies; preserve frozen analysis files separately."""
from pathlib import Path
import argparse,json,urllib.request,zipfile,io,subprocess,sys
ROOT=Path(__file__).resolve().parent
TABLES=['nlst_prsn','nlst_screen','nlst_ctab','nlst_canc']
URLS={name:f'https://idc-open-metadata.s3.amazonaws.com/bigquery_export/idc_v21_clinical/{name}/000000000000.parquet' for name in TABLES}
ORIGINAL={
    'kaggle':'https://www.kaggle.com/api/v1/datasets/download/adnananalytics/lung-cancer-prediction-dataset?datasetVersionNumber=1',
    'matsci':'https://raw.githubusercontent.com/wy314159-lyq/MatSci-ML-Studio1/main/case2.csv'}
def request(url,method='GET'):
    return urllib.request.urlopen(urllib.request.Request(url,method=method,headers={'User-Agent':'IBRB-reproduction/1.0'}),timeout=120)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['original','nlst'],default='original')
    ap.add_argument('--provider',choices=['kaggle','matsci'],default='kaggle')
    ap.add_argument('--output',type=Path,default=ROOT/'downloaded_data')
    ap.add_argument('--check-links',action='store_true');ap.add_argument('--build',action='store_true')
    ap.add_argument('--local-nlst-source',type=Path,help='Use existing four Parquet tables for cohort rebuilding')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    urls=URLS if args.dataset=='nlst' else {args.provider:ORIGINAL[args.provider]}
    if args.check_links:
        records=[]
        for name,url in urls.items():
            try:
                with request(url,'HEAD') as response:
                    records.append(dict(name=name,url=url,status=response.status,content_length=response.headers.get('Content-Length')))
            except Exception as exc:records.append(dict(name=name,url=url,error=str(exc)))
        (args.output/'link_check.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(json.dumps(records,indent=2));return
    if args.dataset=='original':
        import pandas as pd
        import numpy as np
        with request(ORIGINAL[args.provider]) as response:content=response.read()
        if args.provider=='kaggle':
            (args.output/'kaggle_version1.zip').write_bytes(content)
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                file=next(n for n in archive.namelist() if n.lower().endswith('.csv'))
                content=archive.read(file)
        (args.output/'original_public_source.csv').write_bytes(content)
        data=pd.read_csv(io.BytesIO(content));frozen=pd.read_csv(ROOT/'data/original.csv')
        if not pd.api.types.is_numeric_dtype(data.PULMONARY_DISEASE):data['PULMONARY_DISEASE']=data.PULMONARY_DISEASE.map({'YES':1,'NO':0})
        assert list(data.columns)==list(frozen.columns) and len(data)==5000
        discrete=[c for c in frozen if c not in ['ENERGY_LEVEL','OXYGEN_SATURATION']]
        np.testing.assert_array_equal(data[discrete],frozen[discrete])
        np.testing.assert_allclose(data[['ENERGY_LEVEL','OXYGEN_SATURATION']],frozen[['ENERGY_LEVEL','OXYGEN_SATURATION']],atol=5.1e-9,rtol=0)
        data.to_csv(args.output/'original_public_mapped.csv',index=False)
        record=dict(provider=args.provider,source_url=ORIGINAL[args.provider],rows=5000,
            label_mapping='YES->1, NO->0; existing numeric labels preserved',discrete_and_labels_match=True,
            maximum_continuous_difference=float(np.max(np.abs(data[['ENERGY_LEVEL','OXYGEN_SATURATION']].to_numpy()-frozen[['ENERGY_LEVEL','OXYGEN_SATURATION']].to_numpy()))),
            frozen_analysis_file_replaced=False)
        (args.output/'source_comparison.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record,indent=2));return
    source=args.local_nlst_source or args.output/'raw_nlst'
    source.mkdir(parents=True,exist_ok=True)
    if not args.local_nlst_source:
        for name,url in URLS.items():
            with request(url) as response, (source/f'{name}.parquet').open('wb') as target:
                import shutil
                shutil.copyfileobj(response,target)
            print('DOWNLOADED',name,flush=True)
    if args.build:
        cohort=args.output/'rebuilt_nlst'
        subprocess.run([sys.executable,str(ROOT/'build_nlst_cohort.py'),'--source',str(source),'--output',str(cohort)],check=True)
        import pandas as pd
        import numpy as np
        new=pd.read_csv(cohort/'患者级建模候选.csv');old=pd.read_csv(ROOT/'data/nlst.csv')
        assert list(new.columns)==list(old.columns)
        np.testing.assert_allclose(new.to_numpy(float),old.to_numpy(float),atol=0,rtol=0,equal_nan=True)
        record=dict(passed=True,rows=len(new),positive=int(new.label_365d.sum()),unique_patient_ids=new.pid.is_unique,
            frozen_analysis_data_match=True,source='IDC v21 four public clinical tables')
        (args.output/'nlst_rebuild_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record,indent=2))
if __name__=='__main__':main()
