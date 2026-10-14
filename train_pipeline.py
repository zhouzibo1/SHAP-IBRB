"""Regenerate main-method SHAP, nested selection, final fit and held-out evaluation."""
from pathlib import Path
import os,sys,json,argparse,time
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from final_core import fit_soft,soft_predict,restore
from nlst_core import supported_boundaries,rule_fit,predict_plan,plan_for
from upstream_core import shap_fit,fold_ids,folds,upstream,transform
from reproduce import metrics,read

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def original_fit(e,out):
    stage=f"repeat{e['repeat']}_fold{e['fold']}";folder=out/'original'/stage;folder.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'data/original.csv');X=data.iloc[:,:-1].to_numpy(float);y=data.iloc[:,-1].to_numpy(int)
    names=list(data.columns[:-1]);tr=np.asarray(e['training_rows'])-1;te=np.asarray(e['test_rows'])-1
    assert not set(tr)&set(te)
    assignment=fold_ids(y[tr],3,e['seed']+700)
    old_members=pd.read_csv(ROOT/'selection/original'/stage/'内层成员.csv')
    np.testing.assert_array_equal(old_members.inner_fold,assignment+1)
    references=next(r['inner'] for r in read(ROOT/'config/original_inner_reference.json') if r['stage']==stage)
    rows=[];grid=[0.,.01,.05,.2]
    for i in range(3):
        train=tr[assignment!=i];va=tr[assignment==i];base=folder/f'inner{i+1}'
        cols=shap_fit(X[train],y[train],e['seed']+800+i,base/'SHAP')
        ref=references[i]
        np.testing.assert_array_equal(train+1,ref['training_rows']);np.testing.assert_array_equal(va+1,ref['validation_rows'])
        np.testing.assert_array_equal(cols,ref['columns'])
        for lam in grid:
            md=base/f'lambda{lam}';model=fit_soft(X[train][:,cols],y[train],cols,names,e['seed']+900+i,lam,md)
            p,ok=soft_predict(model,X[va][:,cols],cols,names)
            np.savez(md/'validation.npz',source_row_1based=va+1,y=y[va],p=p)
            rows.append(dict(inner_fold=i+1,lambda_value=lam,mse=float(np.mean((p-y[va])**2)),rules=model.L))
            print(f'original {stage} inner{i+1} lambda={lam} done',flush=True)
    search=pd.DataFrame(rows);search.to_csv(folder/'inner_search.csv',index=False)
    frozen=pd.read_csv(ROOT/'selection/original'/stage/'内层搜索.csv')
    merged=search.merge(frozen,on=['inner_fold','lambda_value'],suffixes=('_new','_frozen'))
    error=float(np.max(np.abs(merged.mse_new-merged.mse_frozen)))
    np.testing.assert_allclose(merged.mse_new,merged.mse_frozen,atol=1e-12,rtol=1e-12)
    means=search.groupby('lambda_value').mse.mean();chosen=min(grid,key=lambda v:means.loc[v])
    assert chosen==e['lambda_value']
    cols=shap_fit(X[tr],y[tr],e['seed'],folder/'SHAP')
    np.testing.assert_array_equal(cols,e['columns'])
    model=fit_soft(X[tr][:,cols],y[tr],cols,names,e['seed'],chosen,folder/'final')
    write(folder/'fit_verification.json',dict(task='original',stage=stage,inner_candidates=12,
        SHAP_fits=4,selected_lambda=chosen,maximum_inner_mse_difference=error,
        train_test_disjoint=True,all_training_completed_before_evaluation=True))
    return dict(task='original',stage=stage,folder=str(folder),columns=cols.tolist(),evaluation_ids=(te+1).tolist(),lambda_value=chosen)

def nlst_fit(e,out):
    stage=e['stage'];folder=out/'nlst'/stage;folder.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'data/nlst.csv').set_index('pid');train=data.loc[e['training_pids']]
    y=train.label_365d.to_numpy(int);assignment=folds(y,3,202610070)
    np.testing.assert_array_equal(assignment,e['inner_fold'])
    assert not set(e['training_pids'])&set(e['evaluation_pids'])
    grid=[(m,s) for m in [10,20,40] for s in [.15,.5]];rows=[]
    for i in range(3):
        tr=train.iloc[np.flatnonzero(assignment!=i)];va=train.iloc[np.flatnonzero(assignment==i)]
        base=folder/f'inner{i+1}';X,y,names,cols,prep=upstream(tr,base/'SHAP')
        V=transform(va,prep)[0];vy=va.label_365d.to_numpy(int);selected=[names[j] for j in cols]
        for minimum,sigma in grid:
            md=base/f'min{minimum}_sigma{sigma}';md.mkdir(parents=True,exist_ok=True)
            bounds=[supported_boundaries(X[:,j],y,minimum) for j in cols]
            model,info=rule_fit(X[:,cols],y,selected,20261003+i,True,.05,md,custom_boundaries=bounds,sigma0=sigma)
            p,ok=predict_plan(model,plan_for(V[:,cols],bounds,selected,True))
            np.savez(md/'validation.npz',pid=va.index.to_numpy(),y=vy,p=p)
            rows.append(dict(inner=i+1,candidate=f'{minimum},{sigma}',mse=float(np.mean((p-vy)**2)),n=len(vy)))
            print(f'nlst {stage} inner{i+1} support={minimum} sigma={sigma} done',flush=True)
    search=pd.DataFrame(rows);search.to_csv(folder/'inner_search.csv',index=False)
    frozen=pd.read_csv(ROOT/'selection/nlst'/stage/'内层搜索.csv');frozen=frozen[frozen.family.eq('IBRB')]
    merged=search.merge(frozen,on=['inner','candidate'],suffixes=('_new','_frozen'))
    error=float(np.max(np.abs(merged.mse_new-merged.mse_frozen)))
    np.testing.assert_allclose(merged.mse_new,merged.mse_frozen,atol=1e-12,rtol=1e-12)
    means={f'{m},{s}':float(np.average(search[search.candidate.eq(f'{m},{s}')].mse,weights=search[search.candidate.eq(f'{m},{s}')].n)) for m,s in grid}
    choice=min(means,key=means.get)
    assert choice==read(ROOT/'selection/nlst'/stage/'训练内选择.json')['IBRB']['candidate']
    minimum,sigma=choice.split(',');minimum=int(minimum);sigma=float(sigma)
    X,y,names,cols,prep=upstream(train,folder/'SHAP')
    np.testing.assert_array_equal(cols,e['columns']);selected=[names[j] for j in cols]
    bounds=[supported_boundaries(X[:,j],y,minimum) for j in cols];md=folder/'final';md.mkdir(exist_ok=True)
    model,info=rule_fit(X[:,cols],y,selected,20261003,True,.05,md,custom_boundaries=bounds,sigma0=sigma)
    write(folder/'fit_verification.json',dict(task='nlst',stage=stage,inner_candidates=18,
        SHAP_fits=4,selected_min_child=minimum,selected_sigma=sigma,maximum_inner_mse_difference=error,
        train_test_disjoint=True,all_training_completed_before_evaluation=True))
    return dict(task='nlst',stage=stage,folder=str(folder),columns=cols.tolist(),evaluation_ids=e['evaluation_pids'],preprocessing=prep)

def evaluate(entry):
    folder=Path(entry['folder']);cols=np.asarray(entry['columns']);model=restore(folder/'final')
    if entry['task']=='original':
        data=pd.read_csv(ROOT/'data/original.csv');ids=np.asarray(entry['evaluation_ids'])-1
        X=data.iloc[ids,:-1].to_numpy(float);y=data.iloc[ids,-1].to_numpy(int);names=list(data.columns[:-1])
        p,ok=soft_predict(model,X[:,cols],cols,names)
        saved=pd.read_csv(ROOT/f"predictions/original/{entry['stage']}.csv",float_precision='round_trip')
    else:
        data=pd.read_csv(ROOT/'data/nlst.csv').set_index('pid').loc[entry['evaluation_ids']]
        X,names=transform(data,entry['preprocessing']);y=data.label_365d.to_numpy(int)
        p,ok=predict_plan(model,plan_for(X[:,cols],model.boundaries,[names[j] for j in cols],True))
        saved=pd.read_csv(ROOT/f"predictions/nlst/{entry['stage']}.csv",float_precision='round_trip')
    np.testing.assert_array_equal(y,saved.y_true);np.testing.assert_allclose(p,saved.score,atol=1e-12,rtol=1e-12)
    report=dict(task=entry['task'],stage=entry['stage'],passed=True,
        maximum_prediction_difference=float(np.max(np.abs(p-saved.score))),**metrics(y,p))
    pd.DataFrame(dict(id=entry['evaluation_ids'],y_true=y,score=p,matched=ok)).to_csv(folder/'evaluation_predictions.csv',index=False)
    write(folder/'verification.json',report);return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['original','nlst','both'],default='both')
    ap.add_argument('--stage',default='repeat1_fold1',help='repeat1_fold1..repeat2_fold5 or NLST confirmation')
    ap.add_argument('--all',action='store_true',help='All10 original or all11 NLST stages, or both')
    ap.add_argument('--output',type=Path,default=ROOT/'pipeline_output')
    args=ap.parse_args();start=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True)
    selected=[]
    for task in ['original','nlst']:
        if args.task not in [task,'both']:continue
        for e in read(ROOT/f'config/{task}_models.json'):
            stage=f"repeat{e['repeat']}_fold{e['fold']}" if task=='original' else e['stage']
            if args.all or stage==args.stage:selected.append((task,e))
    if not selected:raise ValueError('Stage is not present for the chosen task')
    write(args.output/'run_plan.json',dict(tasks=[dict(task=t,stage=(f"repeat{e['repeat']}_fold{e['fold']}" if t=='original' else e['stage'])) for t,e in selected],
        mode='frozen-protocol reconstruction',SHAP_refitted=True,inner_search_refitted=True,
        main_method_only=True,outer_evaluation='after all requested fits finish',threads=2))
    fitted=[]
    for task,e in selected:
        fitted.append(original_fit(e,args.output) if task=='original' else nlst_fit(e,args.output))
    write(args.output/'all_fits_ready.json',dict(stages=len(fitted),outer_evaluation_started=False))
    reports=[evaluate(entry) for entry in fitted]
    write(args.output/'completed.json',dict(passed=True,stages=len(reports),reports=reports,
        seconds=time.perf_counter()-start,main_method_only=True))
    print('FULL_PIPELINE_CHECK_COMPLETE',len(reports),flush=True)

if __name__=='__main__':main()
