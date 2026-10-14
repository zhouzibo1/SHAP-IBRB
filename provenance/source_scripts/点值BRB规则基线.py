"""Explicit point-reference BRB/R-NBRB reconstruction for the revised benchmark."""
from pathlib import Path
import sys,json,time
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
for p in [ROOT,HERE,HERE/'依赖']:sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.special import expit
from scipy.stats import rankdata
import 论文配置引擎 as eng
from NLST训练内可行性评估 import score,write_json

def linear_membership(X,refs):
    values=[]
    for j,ref in enumerate(refs):
        x=np.clip(X[:,j],ref[0],ref[-1]);k=len(ref)
        left=np.clip(np.searchsorted(ref,x,side='right')-1,0,k-2)
        fraction=(x-ref[left])/(ref[left+1]-ref[left])
        m=np.zeros((len(X),k));m[np.arange(len(X)),left]=1-fraction;m[np.arange(len(X)),left+1]=fraction
        values.append(m)
    return values

def activation(plan,grid,theta,delta,reliability,operator=None,linear_logs=None):
    relative=delta/delta.max()
    power=relative if operator is None else relative/(1+relative-reliability)
    logs=np.zeros((len(plan[0]),len(grid)))
    for j,m in enumerate(plan):
        if operator is not None:
            m=expit(operator*(m-.5));m=m/m.sum(1,keepdims=True)
        if operator is None and linear_logs is not None:
            log_m=linear_logs[j]
        else:
            with np.errstate(divide='ignore'):log_m=np.log(m)
        logs+=power[j]*log_m[:,grid[:,j]]
    logs+=np.log(theta)[None,:];logs-=logs.max(1,keepdims=True)
    weights=np.exp(logs);weights/=weights.sum(1,keepdims=True)
    return weights

def fuse(weights,beta):
    residual=1-weights
    baseline=np.prod(residual,axis=1)
    positive=np.prod(residual+weights*beta[None,:,1],axis=1)-baseline
    negative=np.prod(residual+weights*beta[None,:,0],axis=1)-baseline
    return positive/(positive+negative)

def project(z,L,M,nonlinear):
    z=np.array(z,copy=True);one=z.ndim==1
    if one:z=z[:,None]
    b=z[:2*L].reshape(L,2,-1)
    first=np.clip((1+b[:,0]-b[:,1])/2,1e-4,1-1e-4)
    b[:,0]=first;b[:,1]=1-first
    z[2*L:]=np.clip(z[2*L:],1e-4,1)
    return z[:,0] if one else z

def unpack(z,L,M,nonlinear):
    beta=z[:2*L].reshape(L,2);theta=z[2*L:3*L];delta=z[3*L:3*L+M]
    operator=1+19*z[-1] if nonlinear else None
    return beta,theta,delta,operator

def fit(X,y,seed,family,folder):
    nonlinear=family=='RNBRB'
    refs=[np.unique(v) if len(np.unique(v))==2 else np.unique(np.quantile(v,[0,.5,1])) for v in X.T]
    assert all(len(r)>=2 for r in refs)
    grid=np.array(list(product(*[range(len(r)) for r in refs])),int);L,M=len(grid),X.shape[1]
    mean,sd=X.mean(0),X.std(0)
    reliability=((X>=mean-3*sd)&(X<=mean+3*sd)).mean(0)
    plan=linear_membership(X,refs);operator=9. if nonlinear else None
    with np.errstate(divide='ignore'):linear_logs=[np.log(m) for m in plan]
    initial_weights=activation(plan,grid,np.full(L,.9),np.ones(M),reliability,operator)
    mass=initial_weights.sum(0);positive=initial_weights.T@y
    p=(positive+1)/(mass+2);beta=np.column_stack([1-p,p])
    x0=np.r_[beta.ravel(),np.full(L,.9),np.ones(M),[8/19] if nonlinear else []]
    def fitness(z):
        beta,theta,delta,a=unpack(z,L,M,nonlinear)
        return float(np.mean((fuse(activation(plan,grid,theta,delta,reliability,a,linear_logs),beta)-y)**2))
    z,info=eng.cmaes_min(x0,fitness,L,seed,generations=200,sigma0=.5,projector=lambda v:project(v,L,M,nonlinear))
    folder.mkdir(parents=True)
    np.savez(folder/'参数.npz',parameters=z,initial_parameters=x0,grid=grid,reliability=reliability,mean=mean,sd=sd)
    write_json(folder/'配置.json',dict(family=family,reference_points=[r.tolist() for r in refs],rules=L,
        M=M,seed=seed,nonlinear=nonlinear,fit_info=info,operator_range=[1,20] if nonlinear else None,
        reference_policy='two observed levels for binary,training min/median/max for other variables',
        initializer='fuzzy rule responsibility Laplace smoothing',tolerance_sigma=3,
        matching='linear triangular and endpoint extension; RNBRB sigmoid normalization',
        objective='raw training MSE',generation_budget=200,sigma0=.5))
    return L,info

def restore(folder):
    cfg=json.loads((folder/'配置.json').read_text(encoding='utf-8'));saved=np.load(folder/'参数.npz')
    return cfg,saved

def predict(folder,X):
    cfg,saved=restore(folder);refs=[np.array(v) for v in cfg['reference_points']]
    beta,theta,delta,a=unpack(saved['parameters'],cfg['rules'],cfg['M'],cfg['nonlinear'])
    weights=activation(linear_membership(X,refs),saved['grid'],theta,delta,saved['reliability'],a)
    return fuse(weights,beta)

def fit_case(ref,family,out):
    data=pd.read_csv(ROOT/'Lung Cancer Dataset.csv')
    X,y=data.iloc[:,:-1].to_numpy(float),data.iloc[:,-1].to_numpy(int)
    tr=np.array(ref['training_rows'])-1;cols=np.array(ref['columns'])
    folder=out/f'repeat{ref["repeat"]}_fold{ref["fold"]}'/family
    start=time.perf_counter();L,info=fit(X[tr][:,cols],y[tr],ref['seed'],family,folder)
    return dict(**{k:v for k,v in ref.items() if k not in ['model','family','model_path']},
        model=family+'1000_same7',family=family,model_path=str(folder),rules=L,fit_seconds=time.perf_counter()-start)

def main():
    prior=Path((HERE/'最新原数据1000交叉验证目录.txt').read_text(encoding='utf-8').strip())
    baseline=Path((HERE/'最新原数据1000传统IBRB目录.txt').read_text(encoding='utf-8').strip())
    out=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/('原数据1000点值规则基线_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=True);(HERE/'最新原数据1000点值规则目录.txt').write_text(str(out),encoding='utf-8')
    data=pd.read_csv(ROOT/'Lung Cancer Dataset.csv');X,y=data.iloc[:,:-1].to_numpy(float),data.iloc[:,-1].to_numpy(int)
    refs=[e for e in json.loads((prior/'模型清单.json').read_text(encoding='utf-8')) if e['family']=='soft']
    if not (out/'运行前固定方案.json').exists():write_json(out/'运行前固定方案.json',dict(reference=str(prior),baseline=str(baseline),training_n=1000,outer_folds=10,
        input_policy='same training SHAP7',rules='Cartesian antecedent reference combinations;binary2,other min/median/max3',
        BRB='linear matching,relative attributeweights,normalized rule activation,complete binary beliefs,analytical ER',
        RNBRB='source equations6-13;sigmoid9initial,center.5,operatoroptimized1-20;reliability train fraction withinmean±3sd',
        constraints='complete binary belief simplex;rule/attributeweights [1e-4,1];sameCMA200sigma.5',
        formula_clarification='eq11 denominator sum_l theta_l*partial_l to meet sumweights1;eq7 counts points inside tolerance',
        reconstruction='new explicitly declared benchmark adaptation; original manuscript point-reference configurations unavailable',
        selection='fixed protocol,no external selection',source='https://pmc.ncbi.nlm.nih.gov/articles/PMC12651893/',
        scope='development reconstruction; no automatic manuscript changes',NLST_test_scored_n=0))
    entries=[];pending=[]
    for ref in refs:
        tr=np.array(ref['training_rows'])-1;cols=np.array(ref['columns'])
        for family in ['BRB','RNBRB']:
            folder=out/f'repeat{ref["repeat"]}_fold{ref["fold"]}'/family
            if (folder/'配置.json').exists():
                saved_entries=json.loads((out/'模型清单.json').read_text(encoding='utf-8'))
                entries.append(next(e for e in saved_entries if e['repeat']==ref['repeat'] and e['fold']==ref['fold'] and e['family']==family))
                continue
            pending.append((ref,family))
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(fit_case,ref,family,out) for ref,family in pending]
        print(f'RESUMED saved{len(entries)} pending{len(pending)} processes3',flush=True)
        for future in as_completed(futures):
            e=future.result();entries.append(e)
            write_json(out/'模型清单.json',entries)
            print(f'SAVED repeat{e["repeat"]} fold{e["fold"]} {e["family"]} rules{e["rules"]} seconds{e["fit_seconds"]:.1f}',flush=True)
    write_json(out/'全部评价前锁定.json',dict(all_models_ready=True,models=20))
    rows=[]
    for e in entries:
        for role,field in [('train','training_rows'),('holdout','test_rows')]:
            idx=np.array(e[field])-1;p=predict(Path(e['model_path']),X[idx][:,e['columns']])
            assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
            path=Path(e['model_path'])/f'{role}_预测.csv'
            pd.DataFrame(dict(source_row_1based=idx+1,y_true=y[idx],score=p)).to_csv(path,index=False)
            metric=score(y[idx],p);rows.append(dict(repeat=e['repeat'],fold=e['fold'],model=e['model'],subset=role,**metric))
    result=pd.DataFrame(rows);result.to_csv(out/'全部指标.csv',index=False)
    old=pd.read_csv(baseline/'全部基线指标.csv');combined=pd.concat([old,result],ignore_index=True)
    combined.to_csv(out/'全部基线指标.csv',index=False)
    combined[combined.subset.eq('holdout')].groupby('model')[['accuracy','auc','mse']].agg(['mean','std']).to_csv(out/'全部基线十折汇总.csv')
    write_json(out/'完成记录.json',dict(completed=True,models=20,automatic_adoption=False))
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
