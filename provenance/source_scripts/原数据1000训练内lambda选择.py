"""Bounded nested lambda selection, reusing identical verified train-only fits."""
from pathlib import Path
import sys,json,time
from datetime import datetime,timezone,timedelta
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
for p in [ROOT,HERE,HERE/'依赖']:sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
from 连续规则软匹配验证 import fit_soft,soft_predict
from 连续软匹配交付核验 import independent_soft
from CMA代数延长小验证 import restore
from NLST训练内可行性评估 import score,write_json
GRID=[0.,.01,.05,.2]

def run_case(ref,out,source):
    started=time.perf_counter();data=pd.read_csv(ROOT/'Lung Cancer Dataset.csv')
    X,y=data.iloc[:,:-1].to_numpy(float),data.iloc[:,-1].to_numpy(int);names=list(data.columns[:-1])
    name=f'repeat{ref["repeat"]}_fold{ref["fold"]}';folder=out/name;folder.mkdir(exist_ok=True)
    prior=source/name;members=pd.read_csv(prior/'内层成员.csv');records=[];new_fits=0
    np.testing.assert_array_equal(members.source_row_1based,ref['training_rows'])
    members.to_csv(folder/'内层成员.csv',index=False)
    for f in [1,2,3]:
        base=prior/f'inner{f}';info=json.loads((base/'成员.json').read_text(encoding='utf-8'))
        tr=np.array(info['training_rows'])-1;va=np.array(info['validation_rows'])-1;cols=np.array(info['columns'])
        assert not set(tr)&set(va) and not set(ref['test_rows'])&set(info['training_rows']+info['validation_rows'])
        write_json(folder/f'inner{f}_来源.json',dict(reference=str(base),**info))
        for lam in GRID:
            reused=lam==.05;md=base/'IG'/'min0' if reused else folder/f'inner{f}'/f'lambda{lam}'
            if (md/'配置.json').exists():model=restore(md)
            else:
                model=fit_soft(X[tr][:,cols],y[tr],cols,names,info['seed'],lam,md);new_fits+=1
            cfg=json.loads((md/'配置.json').read_text(encoding='utf-8'));assert cfg['lambda_value']==lam and cfg['seed']==info['seed']
            p,_=soft_predict(model,X[va][:,cols],cols,names);pred=folder/f'inner{f}_lambda{lam}_预测.csv'
            pd.DataFrame(dict(source_row_1based=va+1,y_true=y[va],score=p)).to_csv(pred,index=False)
            records.append(dict(repeat=ref['repeat'],fold=ref['fold'],inner_fold=f,lambda_value=lam,
                model_path=str(md),prediction_path=str(pred),reused_prior=reused,mse=float(np.mean((p-y[va])**2)),rules=model.L))
        pd.DataFrame(records).to_csv(folder/'内层搜索.csv',index=False)
        print(f'INNER_READY {name} inner{f}',flush=True)
    means=pd.DataFrame(records).groupby('lambda_value').mse.mean();chosen=min(GRID,key=lambda v:means.loc[v])
    choice=dict(repeat=ref['repeat'],fold=ref['fold'],selected_lambda=chosen,inner_mean_MSE={str(v):float(means.loc[v]) for v in GRID})
    write_json(folder/'训练内选择.json',choice)
    tr=np.array(ref['training_rows'])-1;cols=np.array(ref['columns']);md=Path(ref['model_path']) if chosen==.05 else folder/'final'
    if not (md/'配置.json').exists():fit_soft(X[tr][:,cols],y[tr],cols,names,ref['seed'],chosen,md)
    entry=dict(ref,model='IBRB_soft1000_inner_lambda',model_path=str(md),lambda_value=chosen,reused_outer_fixed=chosen==.05)
    write_json(folder/'最终模型.json',entry)
    write_json(folder/'完成记录.json',dict(completed=True,seconds=time.perf_counter()-started,new_inner_fits=new_fits))
    print(f'CASE_READY {name} lambda{chosen}',flush=True)
    return dict(entry=entry,choice=choice,records=records)

def main():
    if len(sys.argv)>1:out=Path(sys.argv[1]);plan=json.loads((out/'运行前固定方案.json').read_text(encoding='utf-8'));prior=Path(plan['reference']);source=Path(plan['inner_reference'])
    else:
        prior=Path((HERE/'最新原数据1000交叉验证目录.txt').read_text(encoding='utf-8').strip())
        source=Path((HERE/'最新离散化最小区间优化目录.txt').read_text(encoding='utf-8').strip())
        out=HERE/('原数据1000训练内lambda选择_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'));out.mkdir()
        write_json(out/'运行前固定方案.json',dict(reference=str(prior),inner_reference=str(source),grid_lambda=GRID,
            outer_training_n=1000,outer_test_n=1000,outer_folds=10,inner_folds=3,
            selection='unweighted mean of three inner validation raw MSE;grid order breaks ties',
            SHAP='quantile alpha.5,200 rounds;refit on each inner training partition;reuse identical verified saved fits',
            discretization='original IG positive gain,max20,parent5;no minimum-child constraint',
            matching='continuous midpoint beta/r/w interpolation,constant endpoint extension;binary hard;ER',
            CMA='200 generations,sigma.5,same paired seed for all lambdas',
            efficiency='reuse 30 identical IG min0 inner lambda.05 fits;reuse final fixed model when selected lambda.05',
            lock='all choices and final models saved before any new outer scoring',
            scope='development nested lambda comparison on previously investigated folds;no new untouched test set',automatic_adoption=False))
    (HERE/'最新原数据1000训练内lambda目录.txt').write_text(str(out),encoding='utf-8')
    refs=[e for e in json.loads((prior/'模型清单.json').read_text(encoding='utf-8')) if e['family']=='soft']
    entries=[];choices=[];records=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run_case,e,out,source) for e in refs]
        for future in as_completed(futures):
            r=future.result();entries.append(r['entry']);choices.append(r['choice']);records.extend(r['records'])
            write_json(out/'模型清单.json',entries);write_json(out/'全部训练内选择.json',choices);pd.DataFrame(records).to_csv(out/'全部内层搜索.csv',index=False)
    write_json(out/'全部评价前锁定.json',dict(models=10,choices=10,inner_candidates=120,all_ready=True))
    data=pd.read_csv(ROOT/'Lung Cancer Dataset.csv');X,y=data.iloc[:,:-1].to_numpy(float),data.iloc[:,-1].to_numpy(int);names=list(data.columns[:-1]);rows=[]
    for e in entries:
        model=restore(Path(e['model_path']));cols=np.array(e['columns']);folder=out/f'repeat{e["repeat"]}_fold{e["fold"]}'
        for role,field in [('train','training_rows'),('holdout','test_rows')]:
            idx=np.array(e[field])-1;p,matched=independent_soft(model,X[idx][:,cols],cols,names)
            pd.DataFrame(dict(source_row_1based=idx+1,y_true=y[idx],score=p,matched=matched)).to_csv(folder/f'{role}_预测.csv',index=False)
            rows.append(dict(repeat=e['repeat'],fold=e['fold'],model=e['model'],subset=role,lambda_value=e['lambda_value'],rules=model.L,**score(y[idx],p)))
    result=pd.DataFrame(rows);result.to_csv(out/'全部指标.csv',index=False)
    fixed=pd.read_csv(prior/'全部指标.csv',float_precision='round_trip');fixed=fixed[fixed.model.eq('IBRB_soft1000')&fixed.subset.eq('holdout')]
    selected=result[result.subset.eq('holdout')];paired=selected.merge(fixed,on=['repeat','fold'],suffixes=('_selected','_fixed'))
    paired.to_csv(out/'固定与内选逐折比较.csv',index=False)
    summary=pd.concat([selected.assign(condition='inner_selected'),fixed.assign(condition='fixed_0.05')]).groupby('condition')[['accuracy','auc','mse']].agg(['mean','std'])
    summary.to_csv(out/'两配置指标汇总.csv');print(summary.to_string(),flush=True)
    write_json(out/'完成记录.json',dict(completed=True,inner_candidates=120,reused_inner_models=30,additional_inner_models=90,
        reused_final_models=sum(e['reused_outer_fixed'] for e in entries),final_models=10,outer_prediction_files=20,automatic_adoption=False))
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
