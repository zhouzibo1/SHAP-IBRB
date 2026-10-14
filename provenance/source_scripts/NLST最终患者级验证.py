"""Frozen NLST pipeline: development nested CV and disjoint1796-patient confirmation."""
from pathlib import Path
import sys,json
from datetime import datetime,timezone,timedelta
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT),str(HERE),str(HERE/'依赖')]
import numpy as np
import pandas as pd
import lightgbm as lgb
import NLST训练内可行性评估 as run
from NLST最小子区间对照 import supported_boundaries
from 最终软匹配全基线统计校准 import logloss
IG_GRID=[(m,s) for m in [10,20,40] for s in [.15,.5]]
LR_GRID=[.1,1.,10.]
TREE_GRID=[dict(max_depth=3,num_leaves=7),dict(max_depth=6,num_leaves=31)]
SEEDS=[20261003,20261004,20261005]
COHORT=ROOT/'数据来源核查'/'NLST患者级候选_20261005'

def folds(y,k,seed):
    rng=np.random.default_rng(seed);a=np.empty(len(y),int)
    for c in [0,1]:
        for f,ids in enumerate(np.array_split(rng.permutation(np.flatnonzero(y==c)),k)):a[ids]=f
    return a

def subsample(frame,n,seed):
    rng=np.random.default_rng(seed);ys=frame.label_365d;pos=int(round(n*ys.mean()))
    return sorted(np.r_[rng.permutation(frame.index[ys.eq(1)])[:pos],rng.permutation(frame.index[ys.eq(0)])[:n-pos]].tolist())

def upstream(frame,folder):
    folder.mkdir(parents=True,exist_ok=True);prep=run.preprocess_fit(frame);X,names=run.transform(frame,prep);y=frame.label_365d.to_numpy(int);mu,sd=X.mean(0),X.std(0);sd[sd==0]=1
    params=dict(objective='binary',seed=SEEDS[0],num_threads=2,verbosity=-1,deterministic=True,force_col_wise=True)
    model=lgb.train(params,lgb.Dataset((X-mu)/sd,label=y,feature_name=names),num_boost_round=200)
    contributions=model.predict((X-mu)/sd,pred_contrib=True);np.testing.assert_allclose(contributions.sum(1),model.predict((X-mu)/sd,raw_score=True),atol=1e-9)
    imp=np.abs(contributions[:,:-1]).mean(0);eligible=np.flatnonzero(X.std(0)>0);selected=eligible[np.argsort(-imp[eligible],kind='stable')[:7]];assert len(selected)==7
    np.savez(folder/'SHAP标准化.npz',mean=mu,sd=sd,selected=selected);(folder/'SHAP模型.txt').write_text(model.model_to_string(),encoding='utf-8');pd.DataFrame(dict(encoded_feature=names,mean_absolute_SHAP=imp,selected=np.isin(np.arange(len(names)),selected))).to_csv(folder/'SHAP排序.csv',index=False)
    run.write_json(folder/'上游拟合.json',dict(training_pids=frame.index.tolist(),preprocessing=prep,all_encoded_names=names,selected_columns=selected.tolist(),selected_features=[names[j] for j in selected],params=params))
    return X,y,names,selected,prep

def tree_fit(X,y,candidate):
    params=dict(objective='binary',learning_rate=.05,min_data_in_leaf=20,lambda_l2=1.,seed=SEEDS[0],num_threads=2,verbosity=-1,deterministic=True,force_col_wise=True);params.update(TREE_GRID[candidate])
    return lgb.train(params,lgb.Dataset(X,label=y),num_boost_round=200)

def rule_savefit(X,y,names,folder,sigma,minimum=None,equal=False,seed=SEEDS[0]):
    folder.mkdir(parents=True,exist_ok=True)
    bounds=[np.linspace(X[:,j].min(),X[:,j].max(),21) if equal else supported_boundaries(X[:,j],y,minimum) for j in range(7)]
    model,info=run.rule_fit(X,y,names,seed,not equal,0 if equal else .05,folder,custom_boundaries=bounds,sigma0=sigma)
    rid,ok=run.engine.match_rules(X,bounds);support=np.bincount(rid.ravel(),minlength=model.L)
    assert ok.all()
    if not equal:assert support.min()>=minimum
    info.update(min_child=minimum,seed=seed,sigma0=sigma,generations=200,training_support=support.tolist(),selected_features=names)
    run.write_json(folder/'训练信息.json',info);return model

def rule_predict(model,X,names,equal=False):return run.predict_plan(model,run.plan_for(X,model.boundaries,names,not equal))

def stage_fit(name,train_ids,evaluation_ids,out):
    folder=out/name;folder.mkdir(exist_ok=True);data=pd.read_csv(COHORT/'患者级建模候选.csv').set_index('pid');train=data.loc[train_ids];fy=train.label_365d.to_numpy(int);assignment=folds(fy,3,202610070);records=[]
    run.write_json(folder/'患者成员.json',dict(training_pids=train_ids,evaluation_pids=evaluation_ids,inner_fold=assignment.tolist(),SHAP_n=1000,IBRB_n=1000))
    for f in range(3):
        tr=train.iloc[np.flatnonzero(assignment!=f)];va=train.iloc[np.flatnonzero(assignment==f)];md=folder/f'inner{f+1}';X,y,names,selected,prep=upstream(tr,md/'SHAP');V=run.transform(va,prep)[0];vy=va.label_365d.to_numpy(int);sn=[names[j] for j in selected]
        for minimum,sigma in IG_GRID:
            d=md/f'IBRB_min{minimum}_sigma{sigma}';model=rule_savefit(X[:,selected],y,sn,d,sigma,minimum,seed=SEEDS[0]+f);p,ok=rule_predict(model,V[:,selected],sn)
            np.savez_compressed(d/'内层预测.npz',pid=va.index.to_numpy(),y=vy,p=p,matched=ok);records.append(dict(inner=f+1,family='IBRB',candidate=f'{minimum},{sigma}',mse=float(np.mean((p-vy)**2)),n=len(vy)))
        for sigma in [.15,.5]:
            d=md/f'equal_sigma{sigma}';model=rule_savefit(X[:,selected],y,sn,d,sigma,equal=True,seed=SEEDS[0]+f);p,ok=rule_predict(model,V[:,selected],sn,True)
            np.savez_compressed(d/'内层预测.npz',pid=va.index.to_numpy(),y=vy,p=p,matched=ok);records.append(dict(inner=f+1,family='equal140',candidate=str(sigma),mse=float(np.mean((p-vy)**2)),n=len(vy)))
        for suffix,cols in [('same7',selected),('all',np.arange(len(names)))]:
            for ridge in LR_GRID:
                lr=run.fit_logistic(X[:,cols],y,ridge);p=run.lr_predict(lr,V[:,cols]);np.savez(md/f'LR_{suffix}_ridge{ridge}.npz',weights=lr[0],mean=lr[1],sd=lr[2],pid=va.index.to_numpy(),y=vy,p=p)
                records.append(dict(inner=f+1,family='LR_'+suffix,candidate=str(ridge),mse=float(np.mean((p-vy)**2)),n=len(vy)))
            for k in range(2):
                model=tree_fit(X[:,cols],y,k);p=model.predict(V[:,cols]);(md/f'LGBM_{suffix}_k{k}.txt').write_text(model.model_to_string(),encoding='utf-8');np.savez(md/f'LGBM_{suffix}_k{k}_预测.npz',pid=va.index.to_numpy(),y=vy,p=p)
                records.append(dict(inner=f+1,family='LGBM_'+suffix,candidate=str(k),mse=float(np.mean((p-vy)**2)),n=len(vy)))
        print(f'INNER_READY {name} fold{f+1}',flush=True)
    search=pd.DataFrame(records);search.to_csv(folder/'内层搜索.csv',index=False);choices={}
    for family in search.family.unique():
        part=search[search.family.eq(family)];order=part.candidate.drop_duplicates().tolist();mean={k:float(np.average(part[part.candidate.eq(k)].mse,weights=part[part.candidate.eq(k)].n)) for k in order};choices[family]=dict(candidate=min(order,key=lambda k:mean[k]),pooled_inner_MSE=mean)
    run.write_json(folder/'训练内选择.json',choices)
    X,y,names,selected,prep=upstream(train,folder/'SHAP');sn=[names[j] for j in selected];entries=[]
    for seed in SEEDS:
        minimum,sigma=choices['IBRB']['candidate'].split(',');d=folder/f'IBRB_seed{seed}';rule_savefit(X[:,selected],y,sn,d,float(sigma),int(minimum),seed=seed)
        entries.append(dict(model='IBRB' if seed==SEEDS[0] else f'IBRB_seed{seed}',family='IBRB',path=str(d),columns=selected.tolist(),seed=seed))
    sigma=float(choices['equal140']['candidate']);d=folder/'equal140';rule_savefit(X[:,selected],y,sn,d,sigma,equal=True);entries.append(dict(model='equal140',family='equal140',path=str(d),columns=selected.tolist(),seed=SEEDS[0]))
    for suffix,cols in [('same7',selected),('all',np.arange(len(names)))]:
        family='LR_'+suffix;ridge=float(choices[family]['candidate']);lr=run.fit_logistic(X[:,cols],y,ridge);d=folder/f'{family}.npz';np.savez(d,weights=lr[0],mean=lr[1],sd=lr[2]);entries.append(dict(model=family,family='LR',path=str(d),columns=cols.tolist(),seed=SEEDS[0],ridge=ridge))
        family='LGBM_'+suffix;k=int(choices[family]['candidate']);model=tree_fit(X[:,cols],y,k);d=folder/f'{family}.txt';d.write_text(model.model_to_string(),encoding='utf-8');entries.append(dict(model=family,family='LGBM',path=str(d),columns=cols.tolist(),seed=SEEDS[0],candidate=k))
    entries.append(dict(model='prevalence',family='constant',path='',columns=[],seed=0,value=float(y.mean())))
    run.write_json(folder/'模型清单.json',entries);print('STAGE_FITTED '+name,flush=True);return name

def evaluate(name,out,development):
    folder=out/name;members=json.loads((folder/'患者成员.json').read_text(encoding='utf-8'));info=json.loads((folder/'SHAP'/'上游拟合.json').read_text(encoding='utf-8'));prep=info['preprocessing'];data=pd.read_csv(COHORT/'患者级建模候选.csv').set_index('pid');rows=[]
    for e in json.loads((folder/'模型清单.json').read_text(encoding='utf-8')):
        if e['family'] in ['IBRB','equal140']:
            z=np.load(Path(e['path'])/'模型.npz');bounds=[np.array(b) for b in json.loads((Path(e['path'])/'边界.json').read_text(encoding='utf-8'))];model=run.engine.IBRBModel(bounds,z['beta'],z['r'],z['w'])
        elif e['family']=='LR':z=np.load(e['path']);model=(z['weights'],z['mean'],z['sd'])
        elif e['family']=='LGBM':model=lgb.Booster(model_str=Path(e['path']).read_text(encoding='utf-8'))
        for role,ids in [('train',members['training_pids']),('evaluation',members['evaluation_pids'])]:
            frame=data.loc[ids];X,names=run.transform(frame,prep);y=frame.label_365d.to_numpy(int);cols=np.array(e['columns'],int);ok=np.ones(len(ids),bool)
            if e['family'] in ['IBRB','equal140']:p,ok=rule_predict(model,X[:,cols],[names[j] for j in cols],e['family']=='equal140')
            elif e['family']=='LR':p=run.lr_predict(model,X[:,cols])
            elif e['family']=='LGBM':p=model.predict(X[:,cols],num_threads=2)
            else:p=np.full(len(ids),e['value'])
            pd.DataFrame(dict(pid=ids,y_true=y,score=p,matched=ok)).to_csv(folder/f'{e["model"]}_{role}_预测.csv',index=False)
            rows.append(dict(stage=name,model=e['model'],seed=e['seed'],subset=role,training_n=1000,positive_training_n=int(data.loc[members['training_pids'],'label_365d'].sum()),rules=model.L if e['family'] in ['IBRB','equal140'] else 0,unmatched_n=int((~ok).sum()),**run.score(y,p),log_loss=logloss(y,p)))
    pd.DataFrame(rows).to_csv(folder/'指标.csv',index=False);return rows

def main():
    data=pd.read_csv(COHORT/'患者级建模候选.csv').set_index('pid');original=pd.read_csv(COHORT/'候选划分成员表.csv');blocks=Path((HERE/'最新NLST新划分验证目录.txt').read_text(encoding='utf-8').strip());old=pd.read_csv(blocks/'新划分成员表.csv');dev=set(original.loc[original.in_SHAP|original.in_validation,'pid'])|set(old.pid);confirmation=set(data.index)-dev;seen=set();audit=[]
    for folder in HERE.glob('NLST*'):
        if folder.is_dir():
            for p in folder.rglob('*预测*.csv'):
                if 'pid' in pd.read_csv(p,nrows=0).columns:seen.update(pd.read_csv(p,usecols=['pid']).pid.tolist());audit.append(str(p))
    assert seen<=dev and len(dev)==5200 and len(confirmation)==1796
    out=HERE/('NLST最终患者级验证_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'));out.mkdir();(HERE/'最新NLST最终患者级验证目录.txt').write_text(str(out),encoding='utf-8')
    run.write_json(out/'运行前最终配置.json',dict(cohort_n=6996,development_n=5200,confirmation_n=1796,confirmation_positive_n=int(data.loc[sorted(confirmation),'label_365d'].sum()),previous_prediction_files_audited=len(audit),unseen_confirmation_overlap=0,
        cohort_task='first lung cancer diagnosis0-365days after baseline CT;eligible baseline noncalcified nodules;patient unit',
        training_budget='same1000 labelled patients for preprocessing,SHAP and each downstream estimator;top7 encoded terms and all encoded input controls',
        SHAP='training-only binary LightGBM200 defaultlr.1leaves31;meanabs raw logodds SHAP;eligible nonconstant inputs;top7',
        rule_grid=IG_GRID,lambda_fixed=.05,CMA_generations=200,optimizer_seeds=SEEDS,primary_optimizer_seed=SEEDS[0],
        inner='3stratifiedfolds;refit upstream each fold;pooled raw Brier choice;exact ties grid order',
        validation='two repeated stratified patient fivefolds in5200 development;training-only subsample1000 each;fold scores developmental;independent1796 confirmation scored once after all fits',
        LR_ridge_grid=LR_GRID,LGBM_grid=TREE_GRID,LGBM_fixed='200trees lr.05 minleaf20 L2=1',traditional_IBRB='same7;20equalwidth intervals each;hard match;CMA200;inner sigma .15/.5;no smoothness',
        threshold=.5,primary_metrics=['auc','mse','average_precision','log_loss'],seed_aggregation='primary fixedseed plus2 separate robustness conditions;no bestseed or ensemble',
        comparison_scope='independent dataset method validation;different inputs and outcome window from original benchmark;not transport of original fitted symptom model',
        confirmation_discipline='complete all selections and fit all models before any new outer or confirmation score;no further tuning after confirmation'))
    pd.DataFrame(dict(pid=sorted(data.index),subset=['development' if pid in dev else 'confirmation' for pid in sorted(data.index)],y_true=data.loc[sorted(data.index),'label_365d'].to_numpy(int))).to_csv(out/'患者用途.csv',index=False);run.write_json(out/'历史预测来源核查.json',dict(files=audit,used_patients=sorted(seen),confirmation_overlap=0))
    development=data.loc[sorted(dev)];tasks=[]
    for r,seed in [(1,202610061),(2,202610062)]:
        assignment=folds(development.label_365d.to_numpy(int),5,seed)
        for f in range(5):
            outer=development.iloc[np.flatnonzero(assignment!=f)];held=development.iloc[np.flatnonzero(assignment==f)].index.tolist();train=subsample(outer,1000,seed+f+100);tasks.append((f'repeat{r}_fold{f+1}',train,held))
    tasks.append(('confirmation',subsample(development,1000,202610063),sorted(confirmation)));run.write_json(out/'预先固定成员.json',[dict(stage=name,training_pids=tr,evaluation_pids=va) for name,tr,va in tasks])
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(stage_fit,name,tr,va,out) for name,tr,va in tasks]
        for future in as_completed(futures):future.result()
    run.write_json(out/'全部外层评价前锁定.json',dict(stages=11,all_selections_and_models_saved=True,confirmation_scored=False))
    rows=[]
    for name,_,_ in tasks:rows.extend(evaluate(name,out,dev))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'全部指标.csv',index=False);metrics[(metrics.subset=='evaluation')&(metrics.stage!='confirmation')].groupby('model')[['accuracy','auc','mse','average_precision','recall','specificity','log_loss','rules']].agg(['mean','std']).to_csv(out/'开发重复五折均值SD.csv')
    metrics[(metrics.subset=='evaluation')&(metrics.stage=='confirmation')].to_csv(out/'最终确认集指标.csv',index=False);run.write_json(out/'完成记录.json',dict(completed=True,stages=11,final_model_conditions=99,prediction_files=198,confirmation_n=1796,confirmation_scored_once=True))
    print(metrics[(metrics.subset=='evaluation')&(metrics.stage=='confirmation')][['model','accuracy','auc','mse','average_precision','recall']].to_string(index=False),flush=True);print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
