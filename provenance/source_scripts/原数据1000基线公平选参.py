"""Train-only LR/LightGBM tuning on the IBRB lambda-selection partitions."""
from pathlib import Path
import sys,json,time
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(HERE),str(ROOT),str(HERE/'依赖'),str(HERE/'基线依赖')]
import numpy as np
import pandas as pd
import lightgbm as lgb
from 重复五折与内层优化 import shap_fit,lr_fit,lr_predict,gb_fit,gb_predict
from NLST训练内可行性评估 import score,write_json
LR_GRID=[.1,1.,10.]
GB_GRID=[(15,1.),(31,1.),(15,5.)]

def save_fit(family,X,y,candidate,seed,stem):
    stem.parent.mkdir(parents=True,exist_ok=True)
    if family=='LR':
        model=lr_fit(X,y,candidate)
        np.savez(stem.with_suffix('.npz'),weights=model[0],mean=model[1],sd=model[2])
    else:
        model=gb_fit(X,y,seed,*candidate)
        stem.with_suffix('.txt').write_text(model[0].model_to_string(),encoding='utf-8')
        np.savez(stem.with_suffix('.npz'),mean=model[1],sd=model[2])
    return model

def predict(family,model,X):return lr_predict(model,X) if family=='LR' else gb_predict(model,X)

def load(family,stem):
    a=np.load(stem.with_suffix('.npz'))
    if family=='LR':return a['weights'],a['mean'],a['sd']
    return lgb.Booster(model_str=stem.with_suffix('.txt').read_text(encoding='utf-8')),a['mean'],a['sd']

def main():
    if len(sys.argv)>1:
        out=Path(sys.argv[1]);evaluate(out);return
    start=time.perf_counter()
    reference=Path((HERE/'最新原数据1000训练内lambda目录.txt').read_text(encoding='utf-8').strip())
    inner_source=Path(json.loads((reference/'运行前固定方案.json').read_text(encoding='utf-8'))['inner_reference'])
    out=HERE/('原数据1000基线公平选参_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'));out.mkdir()
    (HERE/'最新原数据1000基线公平选参目录.txt').write_text(str(out),encoding='utf-8')
    data=pd.read_csv(ROOT/'Lung Cancer Dataset.csv');X=data.iloc[:,:-1].to_numpy(float);y=data.iloc[:,-1].to_numpy(int)
    refs=sorted(json.loads((reference/'模型清单.json').read_text(encoding='utf-8')),key=lambda e:(e['repeat'],e['fold']))
    write_json(out/'运行前固定方案.json',dict(reference=str(reference),inner_source=str(inner_source),LR_GRID=LR_GRID,LGBM_GRID=GB_GRID,
        LGBM_fixed='200trees,lr.05,depth6,minleaf20;leaves/L2 selected',outer_training_n=1000,outer_evaluation_n=1000,
        inner_partitions='identical to IBRB lambda selection',SHAP='fresh quantile alpha.5 200-round fit per inner training;check selected columns against saved IBRB upstream',
        inputs=['same7','all17'],selection='unweighted mean inner3fold raw Brier;grid order breaks ties, same as IBRB',
        outer_SHAP='reuse verified full1000 selected columns',lock='all40 final models saved before outer scoring',scope='previously used benchmark development CV',
        expected_inner_fits=360,expected_final_fits=40,expected_SHAP_refits=30))
    entries=[];records=[];choices=[]
    for ref in refs:
        name=f'repeat{ref["repeat"]}_fold{ref["fold"]}';folder=out/name;folder.mkdir();tr=np.array(ref['training_rows'])-1
        members=pd.read_csv(reference/name/'内层成员.csv');members.to_csv(folder/'内层成员.csv',index=False)
        np.testing.assert_array_equal(members.source_row_1based.to_numpy(),tr+1)
        cases=[]
        for f in range(3):
            info=json.loads((inner_source/name/f'inner{f+1}'/'成员.json').read_text(encoding='utf-8'))
            train=np.array(info['training_rows'])-1;va=np.array(info['validation_rows'])-1
            assert not set(train)&set(va) and set(train)|set(va)==set(tr)
            cols=shap_fit(X[train],y[train],ref['seed']+800+f,folder/f'inner{f+1}'/'SHAP')
            np.testing.assert_array_equal(cols,info['columns'])
            write_json(folder/f'inner{f+1}'/'成员.json',dict(**info,SHAP_seed=ref['seed']+800+f))
            cases.append((f,train,va,cols))
        for suffix in ['same7','all17']:
            for family,grid in [('LR',LR_GRID),('LGBM',GB_GRID)]:
                losses=[]
                for k,candidate in enumerate(grid):
                    foldloss=[]
                    for f,train,va,cols in cases:
                        c=cols if suffix=='same7' else np.arange(X.shape[1]);stem=folder/f'inner{f+1}'/f'{family}_{suffix}_candidate{k}'
                        model=save_fit(family,X[train][:,c],y[train],candidate,ref['seed']+900+f,stem)
                        p=predict(family,model,X[va][:,c]);loss=float(np.mean((p-y[va])**2));foldloss.append(loss)
                        pd.DataFrame(dict(source_row_1based=va+1,y_true=y[va],score=p)).to_csv(stem.parent/(stem.name+'_预测.csv'),index=False)
                        records.append(dict(repeat=ref['repeat'],fold=ref['fold'],inner_fold=f+1,family=family,inputs=suffix,candidate_index=k,candidate=str(candidate),mse=loss,n=len(va),stem=str(stem),columns=json.dumps(c.tolist())))
                    losses.append(float(np.mean(foldloss)))
                k=int(np.argmin(losses));candidate=grid[k];cols=np.array(ref['columns']) if suffix=='same7' else np.arange(X.shape[1]);stem=folder/f'{family}_{suffix}_final'
                save_fit(family,X[tr][:,cols],y[tr],candidate,ref['seed'],stem)
                e=dict(repeat=ref['repeat'],fold=ref['fold'],model=f'{family}1000_{suffix}_inner',family=family,inputs=suffix,columns=cols.tolist(),training_rows=ref['training_rows'],test_rows=ref['test_rows'],stem=str(stem),candidate_index=k,candidate=candidate)
                entries.append(e);choices.append(dict(repeat=ref['repeat'],fold=ref['fold'],family=family,inputs=suffix,chosen_index=k,candidate=candidate,inner_mean_mse=losses))
        write_json(out/'模型清单.json',entries);write_json(out/'全部训练内选择.json',choices);pd.DataFrame(records).to_csv(out/'全部内层搜索.csv',index=False)
        print('FITTED '+name,flush=True)
    write_json(out/'全部评价前锁定.json',dict(final_models=len(entries),inner_fits=len(records),SHAP_refits=30,all_ready=True))
    evaluate(out)
    write_json(out/'完成记录.json',dict(completed=True,models=40,inner_fits=360,SHAP_refits=30,seconds=time.perf_counter()-start))
    print('COMPLETE '+str(out),flush=True)

def evaluate(out):
    assert json.loads((out/'全部评价前锁定.json').read_text(encoding='utf-8'))['all_ready']
    entries=json.loads((out/'模型清单.json').read_text(encoding='utf-8'))
    data=pd.read_csv(ROOT/'Lung Cancer Dataset.csv');X=data.iloc[:,:-1].to_numpy(float);y=data.iloc[:,-1].to_numpy(int)
    result=[]
    for e in entries:
        model=load(e['family'],Path(e['stem']));cols=np.array(e['columns'])
        for subset,field in [('train','training_rows'),('holdout','test_rows')]:
            idx=np.array(e[field])-1;p=predict(e['family'],model,X[idx][:,cols]);folder=Path(e['stem']).parent
            pd.DataFrame(dict(source_row_1based=idx+1,y_true=y[idx],score=p)).to_csv(folder/f'{e["model"]}_{subset}_预测.csv',index=False)
            result.append(dict(repeat=e['repeat'],fold=e['fold'],model=e['model'],subset=subset,**score(y[idx],p)))
    pd.DataFrame(result).to_csv(out/'全部指标.csv',index=False)
    pd.DataFrame(result).query('subset == "holdout"').groupby('model')[['accuracy','auc','mse','recall','average_precision']].agg(['mean','std']).to_csv(out/'四条件十折汇总.csv')
    write_json(out/'完成记录.json',dict(completed=True,models=40,inner_fits=360,SHAP_refits=30))
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
