# -*- coding: utf-8 -*-
"""Continuous midpoint interpolation, retaining supervised intervals and joint CMA/ER."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE/'依赖'))
sys.path.insert(0,str(ROOT))
import json
from datetime import datetime,timedelta,timezone
import numpy as np
import pandas as pd
import 论文配置引擎 as eng
from 最小核验实验 import metrics,write_json
from CMA代数延长小验证 import restore
from 子区间样本约束优化 import cases_from_source
from 相邻规则平滑有限验证 import CONTINUOUS,neighbor_pairs,smoothness
from 重复五折与内层优化 import save_rule
from 三组最小消融 import Tee

GRID=[0.,.01,.05,.2]


def interpolation_plan(X,bounds,columns,names,extended=True):
    left,_=eng.match_rules(X,bounds)
    right=left.copy()
    fraction=np.zeros_like(left,dtype=float)
    matched=np.ones(len(X),dtype=bool)
    offset=0
    for j,b in enumerate(bounds):
        count=len(b)-1
        v=X[:,j]
        continuous=names[columns[j]] in CONTINUOUS
        if continuous and count>1:
            centers=(b[:-1]+b[1:])/2
            idx=np.clip(np.searchsorted(centers,v,side='right')-1,0,count-2)
            fraction[:,j]=np.clip((v-centers[idx])/(centers[idx+1]-centers[idx]),0,1)
            left[:,j]=offset+idx
            right[:,j]=offset+idx+1
        if not (continuous and extended):
            matched&=(v>=b[0])&(v<=b[-1])
        offset+=count
    return left,right,fraction,matched


def soft_scores(model,plan):
    left,right,fraction,matched=plan
    t=fraction[:,:,None]
    beta=(1-t)*model.beta[left]+t*model.beta[right]
    r=(1-fraction)*model.r[left]+fraction*model.r[right]
    w=(1-fraction)*model.w[left]+fraction*model.w[right]
    p=eng.archived.er_rule_batch(r,w,beta)[:,1]
    return np.where(matched&np.isfinite(p),p,0.),matched


def soft_predict(model,X,cols,names,extended=True):
    return soft_scores(model,interpolation_plan(X,model.boundaries,cols,names,extended))


def hard_extended(model,X,cols,names):
    clipped=np.array(X,copy=True)
    for j,b in enumerate(model.boundaries):
        if names[cols[j]] in CONTINUOUS:
            clipped[:,j]=np.clip(clipped[:,j],b[0],b[-1])
    return eng.predict_proba(model,clipped)


def fit_soft(X,y,cols,names,seed,lam,md,custom_boundaries=None):
    initial=eng.build_ibrb(X,y,optimize=False,custom_boundaries=custom_boundaries)
    L=initial.L
    plan=interpolation_plan(X,initial.boundaries,cols,names)
    assert plan[3].all()
    edges=neighbor_pairs(initial.boundaries,cols,names)
    def loss(z):
        model=eng.IBRBModel(initial.boundaries,z[:2*L].reshape(L,2),z[2*L:3*L],z[3*L:])
        p,_=soft_scores(model,plan)
        return float(np.mean((p-y)**2))+lam*smoothness(model.beta,model.r,model.w,edges)
    z,info=eng.cmaes_min(initial.initial_parameters_,loss,L,seed,generations=200,sigma0=.5)
    model=eng.IBRBModel(initial.boundaries,z[:2*L].reshape(L,2),z[2*L:3*L],z[3*L:])
    info['regularized_objective']=info.pop('training_mse')
    info['initial_regularized_objective']=info.pop('initial_training_mse')
    info['trace']=[{('best_regularized_objective' if k=='best_training_mse' else k):v for k,v in row.items()} for row in info['trace']]
    p,_=soft_scores(model,plan)
    p0,_=soft_scores(initial,plan)
    info['training_mse']=float(np.mean((p-y)**2))
    info['initial_training_mse']=float(np.mean((p0-y)**2))
    info['smoothness_penalty']=smoothness(model.beta,model.r,model.w,edges)
    info['initial_smoothness_penalty']=smoothness(initial.beta,initial.r,initial.w,edges)
    info['objective_name']='soft matching training MSE + lambda*continuous neighbor squared parameter differences'
    model.initial_parameters_,model.fit_info_=initial.initial_parameters_,info
    save_rule(md,model,cols,dict(seed=seed,lambda_value=lam,K_max=20,n_min=5,generations=200,sigma0=.5,
        population_multiplier=1,matching='continuous midpoint linear interpolation of beta,r,w; categorical original hard match',
        endpoints='constant extension of continuous first/last midpoint parameters'))
    return model


def main():
    out=HERE/f'连续规则软匹配验证_{datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")}'
    out.mkdir()
    (HERE/'最新连续软匹配验证目录.txt').write_text(str(out),encoding='utf-8')
    sys.stdout=Tee(sys.stdout,out/'运行日志.txt')
    prior=Path((HERE/'最新相邻规则平滑验证目录.txt').read_text(encoding='utf-8'))
    write_json(out/'运行前固定方案.json',dict(grid_lambda=GRID,matching='continuous midpoint interpolation, original categorical hard matching',
        endpoints='constant extension',diagnostics='same-model soft strict-range and saved hard-model endpoint extension',
        gate='fixed three-fold selected raw MSE improves at least2percent against existing selected hard smooth model',
        held_fixed='SHAP1000top7,IBRB260,test130,IG boundaries,K20,parentnmin5,Laplace,joint beta/r/w,CMA200,sigma0.5,originalpopulation,ER',
        reference=str(prior),selection='raw training-internal3fold validation MSE; no outer selection',
        lock='save all selected models before new outer scoring',interpretation='finite exploratory method extension'))
    source=pd.read_csv(ROOT/'Lung Cancer Dataset.csv')
    X,y=source.iloc[:,:-1].to_numpy(float),source.iloc[:,-1].to_numpy(int)
    names=list(source.columns[:-1])
    cases=cases_from_source(source)
    previous=pd.read_csv(prior/'全部内层指标.csv',float_precision='round_trip')
    inner,diagnostics,entries=[],[],[]
    expanded=False
    for index,case in enumerate(cases):
        if index and not expanded:
            break
        name=case['case']
        (out/name).mkdir()
        old_lam=json.loads((prior/name/'内层选择.json').read_text(encoding='utf-8'))['selected_lambda']
        old_mean=float(previous[(previous['case']==name)&(previous['lambda']==old_lam)].mse.mean())
        local=[]
        for fold in range(1,4):
            ref=case['inner_reference']/f'inner{fold}'
            members=pd.read_csv(ref/'成员.csv')
            tr=pd.read_csv(ref/'规则训练成员.csv').source_row_1based.to_numpy(int)-1
            va=members[members.subset=='validation'].source_row_1based.to_numpy(int)-1
            cols=np.array(json.loads((ref/'排序.json').read_text(encoding='utf-8'))['columns'])
            for lam in GRID:
                md=out/name/f'inner{fold}'/f'lambda{lam}'
                model=fit_soft(X[tr][:,cols],y[tr],cols,names,case['inner_seed']+500+fold-1,lam,md)
                p,matched=soft_predict(model,X[va][:,cols],cols,names)
                path=md/'内层预测.csv'
                pd.DataFrame(dict(source_row_1based=va+1,y_true=y[va],raw_score=p,matched=matched)).to_csv(path,index=False)
                row=dict(case=name,inner_fold=fold,lambda_value=lam,model_directory=str(md),prediction_file=str(path),
                    training_mse=model.fit_info_['training_mse'],**metrics(y[va],p))
                inner.append(row);local.append(row)
                p_strict,matched_strict=soft_predict(model,X[va][:,cols],cols,names,extended=False)
                strict_path=md/'软匹配严格范围预测.csv'
                pd.DataFrame(dict(source_row_1based=va+1,y_true=y[va],raw_score=p_strict,matched=matched_strict)).to_csv(strict_path,index=False)
                diagnostics.append(dict(case=name,inner_fold=fold,lambda_value=lam,mode='soft_strict',
                    model_directory=str(md),prediction_file=str(strict_path),**metrics(y[va],p_strict)))
                print(f'INNER {name} f{fold} lambda={lam} MSE={row["mse"]:.5f}',flush=True)
            old_md=prior/name/f'inner{fold}'/f'lambda{old_lam}'
            old_model=restore(old_md)
            p,matched=hard_extended(old_model,X[va][:,cols],cols,names)
            path=out/name/f'inner{fold}'/'硬匹配仅端点延伸预测.csv'
            pd.DataFrame(dict(source_row_1based=va+1,y_true=y[va],raw_score=p,matched=matched)).to_csv(path,index=False)
            diagnostics.append(dict(case=name,inner_fold=fold,lambda_value=old_lam,mode='hard_extended',
                model_directory=str(old_md),prediction_file=str(path),**metrics(y[va],p)))
        means=pd.DataFrame(local).groupby('lambda_value').mse.mean()
        selected=min(GRID,key=lambda v:means.loc[v])
        gain=float(1-means.loc[selected]/old_mean)
        write_json(out/name/'训练内选择.json',dict(selected_lambda=selected,inner_mean_MSE={float(k):float(v) for k,v in means.items()},
            previous_hard_lambda=old_lam,previous_hard_inner_MSE=old_mean,relative_gain=gain))
        if index==0:
            expanded=gain>=.02
            write_json(out/'小验证门槛判断.json',dict(relative_gain=gain,threshold=.02,expanded=expanded))
        if expanded:
            for seed in case['seeds']:
                md=out/name/f'选中模型_seed{seed}'
                fit_soft(X[case['train']][:,case['columns']],y[case['train']],case['columns'],names,seed,selected,md)
                entries.append(dict(case=name,repeat=case['repeat'],fold=case['fold'],seed=seed,lambda_value=selected,
                    columns=case['columns'].tolist(),model_directory=str(md)))
        pd.DataFrame(inner).to_csv(out/'全部内层指标.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(out/'端点与插值诊断.csv',index=False)
    write_json(out/'模型清单.json',entries)
    write_json(out/'全部评价前锁定.json',dict(all_models_ready=True,selected_models=len(entries)))
    rows=[]
    if expanded:
        by_case={c['case']:c for c in cases}
        for entry in entries:
            case=by_case[entry['case']]
            model=restore(Path(entry['model_directory']))
            pt,_=soft_predict(model,X[case['train']][:,case['columns']],case['columns'],names)
            raw=float(np.mean((pt-y[case['train']])**2))
            for subset,ids in [('test',case['test'])]+([('development',case['development'])] if case['development'] is not None else []):
                for label,extended in [('IBRB_soft',True),('IBRB_soft_strict',False)]:
                    p,matched=soft_predict(model,X[ids][:,case['columns']],case['columns'],names,extended)
                    path=out/entry['case']/f'{label}_seed{entry["seed"]}_{subset}_预测.csv'
                    pd.DataFrame(dict(source_row_1based=ids+1,y_true=y[ids],raw_score=p,matched=matched)).to_csv(path,index=False)
                    values=metrics(y[ids],p)
                    rows.append(dict(**entry,model=label,scope='fixed130' if case['repeat']==0 else 'repeated5fold',subset=subset,
                        training_mse=raw,test_minus_train_mse=values['mse']-raw,prediction_file=str(path),**values))
        previous_outer=pd.read_csv(prior/'全部评价指标.csv',float_precision='round_trip')
        rows.extend(previous_outer[previous_outer.model.isin(['IBRB_original','IBRB_smooth_selected','LR260_17','LR260_SHAP7'])].to_dict('records'))
    pd.DataFrame(rows).to_csv(out/'全部评价指标.csv',index=False)
    write_json(out/'完成记录.json',dict(completed=True,expanded=expanded,inner_models=len(inner),diagnostic_rows=len(diagnostics),
        selected_models=len(entries),metric_rows=len(rows)))
    print('EXPANDED',expanded,'INNER',len(inner),'OUTER',len(rows),flush=True)
    if expanded:
        print(pd.DataFrame(rows).groupby(['scope','subset','model'])[['accuracy','mse','auc']].mean().to_string(),flush=True)
    print('COMPLETE '+str(out),flush=True)


if __name__=='__main__':
    main()
