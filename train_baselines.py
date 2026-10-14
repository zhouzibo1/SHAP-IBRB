"""Portable reconstruction of original and NLST baseline training protocols."""
from pathlib import Path
import os,sys,argparse,json,time
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import joblib
import 论文配置引擎 as engine
import tabular_baselines as tab
from upstream_core import shap_fit,fold_ids,folds,upstream,transform
from reproduce import read,metrics
from final_core import save_rule
from nlst_core import rule_fit,plan_for,predict_plan

FAMILIES=['LR','LGBM','DT','SVM','XGB','EBM','DNN','CNN','BRB','RNBRB','equal140']
GRIDS={'LR':[.1,1.,10.],'LGBM':[(15,1.),(31,1.),(15,5.)],
       'DT':[(3,'gini'),(6,'gini'),(3,'entropy'),(6,'entropy')],'SVM':[.1,1.,10.],'XGB':[100,200,400]}
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def tabfit(family,X,y,candidate,seed):
    if family=='LR':return tab.lr_fit(X,y,candidate)
    if family=='LGBM':return tab.gb_fit(X,y,seed,*candidate)
    model=tab.make_model(family,candidate,seed);model.fit(X,y);return model
def tabpredict(family,model,X):
    if family=='LR':return tab.lr_predict(model,X)
    if family=='LGBM':return tab.gb_predict(model,X)
    return model.predict_proba(X)[:,1].astype(np.float64)
def savetab(family,model,path):
    path.mkdir(parents=True,exist_ok=True)
    if family=='LR':np.savez(path/'model.npz',weights=model[0],mean=model[1],sd=model[2])
    elif family=='LGBM':
        (path/'model.txt').write_text(model[0].model_to_string(),encoding='utf-8')
        np.savez(path/'scaler.npz',mean=model[1],sd=model[2])
    else:joblib.dump(model,path/'model.joblib')

def original_stage(e,out,families,inputs):
    stage=f"repeat{e['repeat']}_fold{e['fold']}";folder=out/'original'/stage;folder.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'data/original.csv');X=data.iloc[:,:-1].to_numpy(float);y=data.iloc[:,-1].to_numpy(int)
    tr=np.asarray(e['training_rows'])-1;te=np.asarray(e['test_rows'])-1
    assert not set(tr)&set(te)
    outercols=shap_fit(X[tr],y[tr],e['seed'],folder/'SHAP');np.testing.assert_array_equal(outercols,e['columns'])
    assignment=fold_ids(y[tr],3,e['seed']+700);cases=[]
    for i in range(3):
        train=tr[assignment!=i];va=tr[assignment==i]
        cols=shap_fit(X[train],y[train],e['seed']+800+i,folder/f'inner{i+1}/SHAP')
        cases.append((i,train,va,cols))
    inner=[];fitted=[]
    for suffix in inputs:
        cols=outercols if suffix=='same7' else np.arange(X.shape[1])
        for family in families:
            if suffix=='all17' and family in ['BRB','RNBRB','equal140']:continue
            md=folder/f'{family}_{suffix}'
            setting=None
            if family in GRIDS:
                losses=[]
                for k,candidate in enumerate(GRIDS[family]):
                    local=[]
                    for i,train,va,ic in cases:
                        c=ic if suffix=='same7' else np.arange(X.shape[1])
                        m=tabfit(family,X[train][:,c],y[train],candidate,e['seed']+900+i)
                        p=tabpredict(family,m,X[va][:,c]);loss=float(np.mean((p-y[va])**2));local.append(loss)
                        inner.append(dict(family=family,inputs=suffix,candidate_index=k,inner_fold=i+1,mse=loss))
                    losses.append(float(np.mean(local)))
                k=int(np.argmin(losses));setting=GRIDS[family][k]
                m=tabfit(family,X[tr][:,cols],y[tr],setting,e['seed']);savetab(family,m,md)
                fitted.append(dict(family=family,inputs=suffix,columns=cols.tolist(),model=m,kind='tabular',setting=setting))
            elif family=='EBM':
                m=tabfit(family,X[tr][:,cols],y[tr],None,e['seed']);savetab(family,m,md)
                fitted.append(dict(family=family,inputs=suffix,columns=cols.tolist(),model=m,kind='tabular',setting='package defaults'))
            elif family in ['DNN','CNN']:
                import torch
                import neural_baselines as neural
                torch.set_num_threads(1 if suffix=='same7' else 2)
                torch.use_deterministic_algorithms(True)
                if suffix=='same7':
                    results=[]
                    for decay in [0.,.001]:
                        for i,train,va,ic in cases:
                            path=folder/f'inner{i+1}'/f'{family}_decay{decay}'
                            neural.fit(X[train][:,ic],y[train],family,e['seed']+900+i,decay,[50,100],path)
                            for epoch in [50,100]:
                                m,mean,sd,_=neural.restore(path/f'epoch{epoch}')
                                p=neural.predict(m,X[va][:,ic],mean,sd)
                                results.append(dict(decay=decay,epoch=epoch,inner_fold=i+1,mse=float(np.mean((p-y[va])**2))))
                    means=pd.DataFrame(results).groupby(['decay','epoch']).mse.mean()
                    decay,epoch=min([(d,t) for d in [0.,.001] for t in [50,100]],key=lambda v:means.loc[v])
                    inner.extend(dict(family=family,inputs=suffix,**row) for row in results)
                else:decay,epoch=0.,100
                neural.fit(X[tr][:,cols],y[tr],family,e['seed'],decay,[epoch],md)
                m,mean,sd,_=neural.restore(md/f'epoch{epoch}')
                fitted.append(dict(family=family,inputs=suffix,columns=cols.tolist(),kind='neural',model=(m,mean,sd),setting=dict(decay=decay,epoch=epoch)))
            elif family in ['BRB','RNBRB']:
                import point_baselines as point
                point.fit(X[tr][:,cols],y[tr],e['seed'],family,md)
                fitted.append(dict(family=family,inputs=suffix,columns=cols.tolist(),kind='point',model=md,setting='fixed CMA200 sigma0.5'))
            else:
                m=engine.build_ibrb(X[tr][:,cols],y[tr],seed=e['seed'],method='equal_width',max_intervals=20)
                save_rule(md,m,cols,dict(seed=e['seed'],generations=200,sigma0=.5,K_per_attr=20))
                fitted.append(dict(family=family,inputs=suffix,columns=cols.tolist(),kind='equal',model=m,setting='20 intervals per attribute'))
            print(f'ORIGINAL_FITTED {stage} {family}_{suffix}',flush=True)
    pd.DataFrame(inner).to_csv(folder/'inner_search.csv',index=False)
    # Final prediction evaluation starts after all requested baseline fits for this stage.
    write(folder/'all_fits_ready.json',dict(models=len(fitted),evaluation_started=False))
    saved=pd.read_csv(ROOT/'source_data/predictions/补实验/最终基线统一比较_20261007/22配置原始外层预测.csv',float_precision='round_trip')
    records=[]
    for item in fitted:
        c=item['columns'];family=item['family'];m=item['model']
        if item['kind']=='tabular':p=tabpredict(family,m,X[te][:,c])
        elif item['kind']=='neural':
            import torch,neural_baselines as neural
            torch.set_num_threads(1 if item['inputs']=='same7' else 2)
            p=neural.predict(m[0],X[te][:,c],m[1],m[2])
        elif item['kind']=='point':
            import point_baselines as point
            p=point.predict(m,X[te][:,c])
        else:p=engine.predict_proba(m,X[te][:,c])[0]
        name=f'{family}1000_{item["inputs"]}' if family!='equal140' else 'IBRB_equal140'
        ref=saved[(saved.model==name)&(saved.repeat==e['repeat'])&(saved.fold==e['fold'])].set_index('source_row_1based').loc[te+1]
        error=float(np.max(np.abs(p-ref.score.to_numpy())))
        np.testing.assert_allclose(p,ref.score,atol=1e-7 if item['kind']=='neural' else 1e-12,rtol=0,err_msg=name)
        pd.DataFrame(dict(source_row_1based=te+1,y_true=y[te],score=p)).to_csv(folder/f'{name}_evaluation.csv',index=False)
        records.append(dict(model=name,stage=stage,prediction_max_error=error,setting=item['setting'],**metrics(y[te],p)))
    write(folder/'verification.json',dict(passed=True,task='original',models=records,inner_conditions=len(inner)))
    return records

def nlst_stage(e,out,families,inputs):
    stage=e['stage'];folder=out/'nlst'/stage;folder.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'data/nlst.csv').set_index('pid');train=data.loc[e['training_pids']]
    assignment=folds(train.label_365d.to_numpy(int),3,202610070);cases=[]
    for i in range(3):
        tr=train.iloc[np.flatnonzero(assignment!=i)];va=train.iloc[np.flatnonzero(assignment==i)]
        X,y,names,cols,prep=upstream(tr,folder/f'inner{i+1}/SHAP');V=transform(va,prep)[0]
        cases.append((i,X,y,V,va.label_365d.to_numpy(int),cols,names))
    X,y,names,cols,prep=upstream(train,folder/'SHAP');fitted=[];inner=[]
    for suffix in inputs:
        c=cols if suffix=='same7' else np.arange(X.shape[1])
        for family in families:
            if family not in ['LR','LGBM','equal140','prevalence']:continue
            if family in ['equal140','prevalence'] and suffix=='all17':continue
            if family=='prevalence':fitted.append((family,suffix,c,float(y.mean()),None));continue
            grid=[.1,1.,10.] if family=='LR' else [0,1] if family=='LGBM' else [.15,.5]
            losses=[]
            for candidate in grid:
                local=[];weights=[]
                for i,A,b,V,vy,ic,inames in cases:
                    cc=ic if suffix=='same7' else np.arange(A.shape[1])
                    if family=='LR':m=tab.fit_logistic(A[:,cc],b,candidate);p=tab.lr_predict(m,V[:,cc])
                    elif family=='LGBM':m=tab.tree_fit(A[:,cc],b,candidate);p=m.predict(V[:,cc])
                    else:
                        bounds=[np.linspace(A[:,j].min(),A[:,j].max(),21) for j in cc]
                        md=folder/f'inner{i+1}/equal_sigma{candidate}';md.mkdir(parents=True,exist_ok=True)
                        m,_=rule_fit(A[:,cc],b,[inames[j] for j in cc],20261003+i,False,0,md,custom_boundaries=bounds,sigma0=candidate)
                        p=predict_plan(m,plan_for(V[:,cc],bounds,[inames[j] for j in cc],False))[0]
                    loss=float(np.mean((p-vy)**2));local.append(loss);weights.append(len(vy))
                    inner.append(dict(family=family,inputs=suffix,inner=i+1,candidate=str(candidate),mse=loss,n=len(vy)))
                losses.append(float(np.average(local,weights=weights)))
            chosen=grid[int(np.argmin(losses))];md=folder/f'{family}_{suffix}';md.mkdir(exist_ok=True)
            if family=='LR':m=tab.fit_logistic(X[:,c],y,chosen);savetab('LR',m,md)
            elif family=='LGBM':m=tab.tree_fit(X[:,c],y,chosen);(md/'model.txt').write_text(m.model_to_string(),encoding='utf-8')
            else:
                bounds=[np.linspace(X[:,j].min(),X[:,j].max(),21) for j in c]
                m,_=rule_fit(X[:,c],y,[names[j] for j in c],20261003,False,0,md,custom_boundaries=bounds,sigma0=chosen)
            fitted.append((family,suffix,c,m,chosen));print(f'NLST_FITTED {stage} {family}_{suffix}',flush=True)
    pd.DataFrame(inner).to_csv(folder/'inner_search.csv',index=False)
    write(folder/'all_fits_ready.json',dict(models=len(fitted),evaluation_started=False))
    evaluation=data.loc[e['evaluation_pids']];V,vnames=transform(evaluation,prep);vy=evaluation.label_365d.to_numpy(int);records=[]
    for family,suffix,c,m,choice in fitted:
        if family=='LR':p=tab.lr_predict(m,V[:,c])
        elif family=='LGBM':p=m.predict(V[:,c])
        elif family=='prevalence':p=np.full(len(vy),m)
        else:p=predict_plan(m,plan_for(V[:,c],m.boundaries,[names[j] for j in c],False))[0]
        name=family+'_'+('all' if suffix=='all17' else suffix) if family in ['LR','LGBM'] else family
        # Frozen baseline probabilities are retained for confirmation, the principal external comparison.
        error=None
        if stage=='confirmation':
            ref=pd.read_csv(ROOT/f'predictions/nlst_confirmation_all/{name}_evaluation_预测.csv',float_precision='round_trip')
            np.testing.assert_allclose(p,ref.score,atol=1e-12,rtol=0,err_msg=name);error=float(np.max(np.abs(p-ref.score)))
        pd.DataFrame(dict(pid=e['evaluation_pids'],y_true=vy,score=p)).to_csv(folder/f'{name}_evaluation.csv',index=False)
        records.append(dict(model=name,stage=stage,setting=choice,prediction_max_error=error,**metrics(vy,p)))
    write(folder/'verification.json',dict(passed=True,task='nlst',models=records,inner_conditions=len(inner)))
    return records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['original','nlst'],default='original')
    ap.add_argument('--stage',default='repeat1_fold1');ap.add_argument('--all',action='store_true')
    ap.add_argument('--families',default='all');ap.add_argument('--inputs',choices=['same7','all17','all'],default='same7')
    ap.add_argument('--output',type=Path,default=ROOT/'baseline_training_output');args=ap.parse_args()
    if args.families=='all':families=FAMILIES if args.task=='original' else ['LR','LGBM','equal140','prevalence']
    else:families=args.families.split(',')
    allowed=FAMILIES if args.task=='original' else ['LR','LGBM','equal140','prevalence']
    if any(f not in allowed for f in families):raise ValueError('Unsupported baseline family for this task')
    inputs=['same7','all17'] if args.inputs=='all' else [args.inputs];records=[];start=time.perf_counter()
    if any(f in ['DNN','CNN'] for f in families):
        import torch
        torch.set_num_interop_threads(1)
    selected=[]
    for e in read(ROOT/f'config/{args.task}_models.json'):
        stage=f"repeat{e['repeat']}_fold{e['fold']}" if args.task=='original' else e['stage']
        if args.all or stage==args.stage:selected.append(e)
    if not selected:raise ValueError('Requested stage is absent')
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'run_plan.json',dict(task=args.task,families=families,inputs=inputs,stages=len(selected),selection='training-only, original method-specific protocol'))
    for e in selected:records.extend(original_stage(e,args.output,families,inputs) if args.task=='original' else nlst_stage(e,args.output,families,inputs))
    pd.DataFrame(records).to_csv(args.output/'metrics.csv',index=False)
    write(args.output/'completed.json',dict(passed=True,models=len(records),seconds=time.perf_counter()-start,results=records))
    print('BASELINES_COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
