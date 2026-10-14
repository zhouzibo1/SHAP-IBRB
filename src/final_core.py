from pathlib import Path
import json
import numpy as np
import 论文配置引擎 as eng
CONTINUOUS = {'AGE','ENERGY_LEVEL','OXYGEN_SATURATION'}
def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


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

def neighbor_pairs(boundaries,columns,names):
    pairs,offset = [],0
    for j,bounds in enumerate(boundaries):
        size = len(bounds)-1
        if names[columns[j]] in CONTINUOUS:
            pairs.extend((offset+k,offset+k+1) for k in range(size-1))
        offset += size
    return np.asarray(pairs,dtype=int).reshape(-1,2)

def smoothness(beta,r,w,pairs):
    if not len(pairs):
        return 0.
    theta = np.column_stack([beta,r,w])
    return float(np.mean((theta[pairs[:,0]]-theta[pairs[:,1]])**2))

def save_rule(directory, model, columns, settings):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory/'模型.npz', beta=model.beta, r=model.r, w=model.w,
             initial_parameters=model.initial_parameters_)
    write_json(directory/'边界.json', [b.tolist() for b in model.boundaries])
    write_json(directory/'配置.json', {'columns': columns.tolist(), **settings,
                                    'fit_info': model.fit_info_})

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

def restore(directory):
    p = np.load(directory/'模型.npz')
    bounds = [np.array(b) for b in json.loads((directory/'边界.json').read_text(encoding='utf-8'))]
    return eng.IBRBModel(bounds, p['beta'], p['r'], p['w'])
