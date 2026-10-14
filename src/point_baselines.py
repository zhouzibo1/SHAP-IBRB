from pathlib import Path
import json
import numpy as np
import pandas as pd
def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
from itertools import product
from scipy.special import expit
import 论文配置引擎 as eng

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
    folder.mkdir(parents=True,exist_ok=True)
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
