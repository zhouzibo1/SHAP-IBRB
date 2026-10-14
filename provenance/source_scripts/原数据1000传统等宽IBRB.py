"""Original20 intervals per attribute hard-matching IBRB, paired repeated CV."""
from pathlib import Path
import sys,json
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
for p in [ROOT,HERE,HERE/'依赖']:sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import 论文配置引擎 as eng
from 重复五折与内层优化 import save_rule
from CMA代数延长小验证 import restore
from NLST训练内可行性评估 import score,write_json

def main():
    prior=Path((HERE/'最新原数据1000交叉验证目录.txt').read_text(encoding='utf-8').strip())
    baseline=Path((HERE/'最新原数据1000深度基线目录.txt').read_text(encoding='utf-8').strip())
    out=HERE/('原数据1000传统等宽IBRB_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    out.mkdir();(HERE/'最新原数据1000传统IBRB目录.txt').write_text(str(out),encoding='utf-8')
    data=pd.read_csv(ROOT/'Lung Cancer Dataset.csv');X,y=data.iloc[:,:-1].to_numpy(float),data.iloc[:,-1].to_numpy(int)
    refs=[e for e in json.loads((prior/'模型清单.json').read_text(encoding='utf-8')) if e['family']=='soft']
    write_json(out/'运行前固定方案.json',dict(reference=str(prior),baseline_reference=str(baseline),
        training_n=1000,outer_folds=10,group='IBRB_equal140',inputs='same training-only SHAP7',
        settings='20 equal-width intervals each,hard strict-range matching,ER,Laplace,r.9,w1/L,CMA200sigma.5 raw training MSE',
        rationale='original traditional IBRB condition; separate from matched-count equalwidth ablation',
        selection='one predefined paired seed eachfold,no tuning',
        scope='development repeated CV extension; no automatic adoption',NLST_test_scored_n=0))
    entries=[]
    for ref in refs:
        tr=np.array(ref['training_rows'])-1;te=np.array(ref['test_rows'])-1;cols=np.array(ref['columns'])
        assert not np.intersect1d(tr,te).size
        folder=out/f'repeat{ref["repeat"]}_fold{ref["fold"]}'
        model=eng.build_ibrb(X[tr][:,cols],y[tr],seed=ref['seed'],method='equal_width',max_intervals=20)
        assert model.L==140
        save_rule(folder,model,cols,dict(seed=ref['seed'],generations=200,sigma0=.5,K_per_attr=20,
            matching='hard strict-range',lambda_value=0.,training_n=1000))
        entries.append(dict(**{k:v for k,v in ref.items() if k not in ['model','family','model_path']},
            model='IBRB_equal140',model_path=str(folder)))
        write_json(out/'模型清单.json',entries)
        print(f'FITTED repeat{ref["repeat"]} fold{ref["fold"]} traditional140',flush=True)
    write_json(out/'全部评价前锁定.json',dict(all_models_saved=True,models=10))
    rows=[];complexity=[]
    for e in entries:
        folder=Path(e['model_path']);model=restore(folder);cols=np.array(e['columns']);tr=np.array(e['training_rows'])-1
        initial=eng.build_ibrb(X[tr][:,cols],y[tr],method='equal_width',max_intervals=20,optimize=False)
        for a,b in zip(model.boundaries,initial.boundaries):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(np.load(folder/'模型.npz')['initial_parameters'],initial.initial_parameters_)
        ids,matched=eng.match_rules(X[tr][:,cols],model.boundaries)
        active=np.unique(ids[ids>=0])
        cfg=json.loads((folder/'配置.json').read_text(encoding='utf-8'))
        assert cfg['fit_info']['dimension']==560 and cfg['fit_info']['generations']==200
        population=10+int(3*np.log(560));assert cfg['fit_info']['candidate_evaluations']==200*population
        complexity.append(dict(repeat=e['repeat'],fold=e['fold'],rules=140,training_active_rules=len(active),
            dimension=560,population=population,candidate_evaluations=200*population))
        for role,field in [('train','training_rows'),('holdout','test_rows')]:
            idx=np.array(e[field])-1;p,matched=eng.predict_proba(model,X[idx][:,cols])
            path=folder/f'{role}_预测.csv';pd.DataFrame(dict(source_row_1based=idx+1,y_true=y[idx],score=p,matched=matched)).to_csv(path,index=False)
            pred=pd.read_csv(path,float_precision='round_trip');metric=score(y[idx],p)
            pos=y[idx].sum();neg=len(idx)-pos;auc=(rankdata(pred.score)[y[idx]==1].sum()-pos*(pos+1)/2)/(pos*neg)
            np.testing.assert_allclose([auc,np.mean((pred.score-y[idx])**2)],[metric['auc'],metric['mse']],atol=1e-12)
            if role=='train':np.testing.assert_allclose(cfg['fit_info']['training_mse'],metric['mse'],atol=1e-12)
            rows.append(dict(repeat=e['repeat'],fold=e['fold'],model=e['model'],subset=role,unmatched_n=int((~matched).sum()),**metric))
    result=pd.DataFrame(rows);result.to_csv(out/'全部指标.csv',index=False)
    pd.DataFrame(complexity).to_csv(out/'实际规则复杂度.csv',index=False)
    old=pd.read_csv(baseline/'全部基线指标.csv');combined=pd.concat([old,result],ignore_index=True)
    combined.to_csv(out/'全部基线指标.csv',index=False)
    combined[combined.subset.eq('holdout')].groupby('model')[['accuracy','auc','mse']].agg(['mean','std']).to_csv(out/'全部基线十折汇总.csv')
    write_json(out/'独立交付核验.json',dict(passed=True,models=10,prediction_files=20,
        equalwidth_boundaries_initialization_and_optimizer_budget_verified=True,raw_mse_and_rank_auc_verified=True,NLST_test_scored_n=0))
    write_json(out/'完成记录.json',dict(completed=True,models=10,automatic_adoption=False))
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
