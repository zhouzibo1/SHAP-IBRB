"""Training-only parameter selection for missing original manuscript baselines."""
from pathlib import Path
import sys,json,time
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
for path in [ROOT,HERE,HERE/'依赖',HERE/'基线依赖']:sys.path.insert(0,str(path))
import numpy as np
import pandas as pd
import sklearn,xgboost,interpret,joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import rankdata
from 重复五折与内层优化 import fold_ids,shap_fit
from NLST训练内可行性评估 import score,write_json

GRIDS={'DT':[(3,'gini'),(6,'gini'),(3,'entropy'),(6,'entropy')],
       'SVM':[.1,1.,10.],'XGB':[100,200,400]}

def make_model(family,candidate,seed):
    if family=='DT':return DecisionTreeClassifier(max_depth=candidate[0],criterion=candidate[1],random_state=seed)
    if family=='SVM':
        base=make_pipeline(StandardScaler(),SVC(kernel='linear',C=candidate,random_state=seed))
        return CalibratedClassifierCV(base,method='sigmoid',
            cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed),ensemble=False,n_jobs=1)
    if family=='XGB':return XGBClassifier(n_estimators=candidate,max_depth=6,learning_rate=.05,
        objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=2,random_state=seed)
    return ExplainableBoostingClassifier(n_jobs=1,random_state=seed)

def main():
    prior=Path((HERE/'最新原数据1000交叉验证目录.txt').read_text(encoding='utf-8').strip())
    out=HERE/('原数据1000四项基线_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    out.mkdir();(HERE/'最新原数据1000四项基线目录.txt').write_text(str(out),encoding='utf-8')
    data=pd.read_csv(ROOT/'Lung Cancer Dataset.csv')
    X,y=data.iloc[:,:-1].to_numpy(float),data.iloc[:,-1].to_numpy(int)
    names=list(data.columns[:-1])
    refs=[e for e in json.loads((prior/'模型清单.json').read_text(encoding='utf-8')) if e['family']=='soft']
    write_json(out/'运行前固定方案.json',dict(reference=str(prior),training_n=1000,outer_test_n=1000,outer_folds=10,
        grids=GRIDS,selection='mean inner3fold raw probability MSE; first grid item breaks ties',
        input_conditions=['training-only SHAP top7','all17'],
        SHAP_inner='refit quantile SHAP on each inner training fold before same7 baseline fitting',
        SVM='linear SVC with train-internal3fold sigmoid probability calibration; scaler inside pipeline; threshold.5',
        EBM='package defaults except random_state and n_jobs1; default internal validation and bagging stay within1000',
        versions=dict(sklearn=sklearn.__version__,xgboost=xgboost.__version__,interpret=interpret.__version__),
        expected_final_models=80,expected_inner_candidates=600,
        scope='development repeated CV; selected new baselines versus previously fixed IBRB/LR/LightGBM',
        lock='all models and choices saved before outer scoring',automatic_adoption=False,NLST_test_scored_n=0))
    entries=[];inner_records=[];choices=[]
    for ref in refs:
        repeat,fold,seed=ref['repeat'],ref['fold'],ref['seed']
        tr=np.array(ref['training_rows'])-1;te=np.array(ref['test_rows'])-1;finalcols=np.array(ref['columns'])
        folder=out/f'repeat{repeat}_fold{fold}';folder.mkdir()
        inner=fold_ids(y[tr],3,seed+700)
        pd.DataFrame(dict(source_row_1based=tr+1,inner_fold=inner+1)).to_csv(folder/'内层成员.csv',index=False)
        inner_cases=[]
        for f in range(3):
            train,va=tr[inner!=f],tr[inner==f]
            cols=shap_fit(X[train],y[train],seed+800+f,folder/f'inner{f+1}'/'SHAP')
            inner_cases.append((f,train,va,cols))
        for suffix in ['same7','all17']:
            selected={}
            for family,grid in GRIDS.items():
                means=[]
                for index,candidate in enumerate(grid):
                    losses=[]
                    for f,train,va,innercols in inner_cases:
                        c=innercols if suffix=='same7' else np.arange(len(names))
                        model=make_model(family,candidate,seed+900+f)
                        model.fit(X[train][:,c],y[train])
                        p=model.predict_proba(X[va][:,c])[:,1].astype(np.float64)
                        loss=float(np.mean((p-y[va])**2));losses.append(loss)
                        d=folder/f'inner{f+1}'
                        pd.DataFrame(dict(source_row_1based=va+1,y_true=y[va],score=p)).to_csv(d/f'{family}_{suffix}_candidate{index}_预测.csv',index=False)
                        inner_records.append(dict(repeat=repeat,fold=fold,inner_fold=f+1,family=family,
                            inputs=suffix,candidate_index=index,candidate=str(candidate),mse=loss))
                    means.append(float(np.mean(losses)))
                best=int(np.argmin(means));selected[family]=grid[best]
                choices.append(dict(repeat=repeat,fold=fold,family=family,inputs=suffix,
                    selected_candidate_index=best,selected_candidate=str(grid[best]),inner_mean_mse=means))
            cols=finalcols if suffix=='same7' else np.arange(len(names))
            for family in ['DT','SVM','XGB','EBM']:
                start=time.perf_counter()
                print(f'START repeat{repeat} fold{fold} {family}_{suffix}',flush=True)
                model=make_model(family,selected.get(family),seed)
                model.fit(X[tr][:,cols],y[tr])
                path=folder/f'{family}_{suffix}.joblib';joblib.dump(model,path)
                if family=='XGB':model.save_model(folder/f'{family}_{suffix}.json')
                if family=='EBM':write_json(folder/f'EBM_{suffix}_参数.json',model.get_params())
                entries.append(dict(repeat=repeat,fold=fold,seed=seed,model=f'{family}1000_{suffix}',family=family,
                    inputs=suffix,columns=cols.tolist(),training_rows=(tr+1).tolist(),test_rows=(te+1).tolist(),
                    model_path=str(path),candidate=str(selected.get(family,'defaults')),fit_seconds=time.perf_counter()-start))
                print(f'SAVED repeat{repeat} fold{fold} {family}_{suffix} seconds{time.perf_counter()-start:.1f}',flush=True)
        write_json(out/'模型清单.json',entries);write_json(out/'全部训练内选择.json',choices)
        pd.DataFrame(inner_records).to_csv(out/'全部内层搜索.csv',index=False)
    assert len(entries)==80 and len(inner_records)==600
    write_json(out/'全部评价前锁定.json',dict(all_models_ready=True,models=80,choices=60))
    rows=[]
    for e in entries:
        model=joblib.load(e['model_path']);cols=np.array(e['columns'])
        for subset,field in [('train','training_rows'),('holdout','test_rows')]:
            idx=np.array(e[field])-1;p=model.predict_proba(X[idx][:,cols])[:,1].astype(np.float64)
            assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
            path=out/f'repeat{e["repeat"]}_fold{e["fold"]}'/f'{e["model"]}_{subset}_预测.csv'
            pd.DataFrame(dict(source_row_1based=idx+1,y_true=y[idx],score=p)).to_csv(path,index=False)
            pred=pd.read_csv(path,float_precision='round_trip');metric=score(y[idx],p)
            pos=y[idx].sum();neg=len(idx)-pos
            auc=(rankdata(pred.score)[y[idx]==1].sum()-pos*(pos+1)/2)/(pos*neg)
            np.testing.assert_allclose([auc,np.mean((pred.score-y[idx])**2)],[metric['auc'],metric['mse']],atol=1e-12)
            rows.append(dict(repeat=e['repeat'],fold=e['fold'],model=e['model'],subset=subset,**metric))
    result=pd.DataFrame(rows);result.to_csv(out/'全部指标.csv',index=False)
    old=pd.read_csv(prior/'全部指标.csv')
    combined=pd.concat([old[result.columns],result],ignore_index=True)
    combined.to_csv(out/'含既有基线全部指标.csv',index=False)
    hold=combined[combined.subset.eq('holdout')]
    hold.groupby('model')[['accuracy','auc','mse','recall','average_precision']].agg(['mean','std']).to_csv(out/'全部基线十折汇总.csv')
    write_json(out/'独立交付核验.json',dict(passed=True,final_models=80,inner_candidate_fits=600,prediction_files=160,
        raw_mse_and_rank_auc_verified=True,NLST_test_scored_n=0))
    write_json(out/'完成记录.json',dict(completed=True,models=80,inner_candidate_fits=600,SHAP_inner_fits=30,automatic_adoption=False))
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
