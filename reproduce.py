"""Portable saved-model prediction and result reconstruction."""
from pathlib import Path
import argparse, json, sys, os
# Original numerical runtime recorded two BLAS/OpenMP threads.
# Set this before importing NumPy so CMA eigendecomposition follows that runtime.
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from final_core import restore, soft_predict, fit_soft

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=.5
    tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)))
    fn=int(np.sum(~pred&(y==1)));tn=int(np.sum(~pred&(y==0)))
    clipped=np.clip(p,np.finfo(float).eps,1-np.finfo(float).eps)
    return dict(n=len(y),accuracy=(tp+tn)/len(y),mse=float(np.mean((p-y)**2)),
        auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),
        recall=tp/(tp+fn),specificity=tn/(tn+fp),f1=2*tp/(2*tp+fp+fn),
        TP=tp,FP=fp,FN=fn,TN=tn,predicted_positive_n=tp+fp,
        log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))))

def compare(row,reference):
    for key,value in row.items():
        if key in reference:
            np.testing.assert_allclose(value,reference[key],atol=2e-12,rtol=2e-12,err_msg=key)

def transform(frame,prep):
    vals=[];names=[]
    for col in ['age','baseline_nodule_count','index_long_diameter_mm','index_perpendicular_diameter_mm','male','cigsmok']:
        vals.append(frame[col].fillna(prep['impute'][col]).to_numpy(float));names.append(col)
    for col in ['race','index_lobe','index_margin','index_attenuation']:
        raw=frame[col].fillna(prep['impute'][col]).to_numpy(float)
        for level in prep['categories'][col]:
            vals.append((raw==level).astype(float));names.append(f'{col}_level_{int(level)}')
    return np.column_stack(vals),names

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    pred=pd.read_csv(ROOT/'source_data/predictions/补实验/最终基线统一比较_20261007/22配置原始外层预测.csv',float_precision='round_trip')
    ref=pd.read_csv(ROOT/'source_data/statistics/补实验/最终基线统一比较_20261007/22配置逐折指标.csv',float_precision='round_trip')
    rows=[]
    for (model,repeat,fold),g in pred.groupby(['model','repeat','fold']):
        row=metrics(g.y_true,g.score)
        reference=ref[(ref.model==model)&(ref.repeat==repeat)&(ref.fold==fold)].iloc[0]
        compare(row,reference);rows.append(dict(model=model,repeat=repeat,fold=fold,**row))
    pd.DataFrame(rows).to_csv(out/'original_220_metrics.csv',index=False)
    pd.DataFrame(rows).groupby('model')[['accuracy','auc','mse']].agg(['mean','std']).to_csv(out/'original_mean_sd.csv')
    data=pd.read_csv(ROOT/'data/original.csv',float_precision='round_trip')
    X=data.iloc[:,:-1].to_numpy(float);y=data.iloc[:,-1].to_numpy(int);names=list(data.columns[:-1])
    errors=[];modelrows=[]
    for e in read(ROOT/'config/original_models.json'):
        tr=np.asarray(e['training_rows'])-1;te=np.asarray(e['test_rows'])-1;cols=np.asarray(e['columns'])
        assert len(tr)==1000 and len(te)==1000 and not set(tr)&set(te)
        model=restore(ROOT/e['model_path']);p,matched=soft_predict(model,X[te][:,cols],cols,names)
        saved=pd.read_csv(ROOT/e['prediction_path'],float_precision='round_trip')
        np.testing.assert_array_equal(saved.source_row_1based,te+1);np.testing.assert_array_equal(saved.y_true,y[te])
        np.testing.assert_allclose(p,saved.score,atol=1e-12,rtol=1e-12)
        error=float(np.max(np.abs(p-saved.score)));errors.append(error)
        modelrows.append(dict(task='original',stage=f"repeat{e['repeat']}_fold{e['fold']}",max_error=error,**metrics(y[te],p)))
    nlst=pd.read_csv(ROOT/'data/nlst.csv',float_precision='round_trip').set_index('pid')
    continuous={'age','baseline_nodule_count','index_long_diameter_mm','index_perpendicular_diameter_mm'}
    development=set();confirmation=set()
    for e in read(ROOT/'config/nlst_models.json'):
        tr=e['training_pids'];te=e['evaluation_pids'];assert len(tr)==1000 and not set(tr)&set(te)
        if e['stage']=='confirmation':confirmation.update(te)
        else:development.update(tr);development.update(te)
        frame=nlst.loc[te];z,names=transform(frame,e['preprocessing']);assert names==e['all_encoded_names']
        cols=np.asarray(e['columns']);model=restore(ROOT/e['model_path'])
        # Use the same midpoint interpolation and ER implementation with task-specific continuous names.
        import final_core
        previous=final_core.CONTINUOUS
        final_core.CONTINUOUS=continuous
        try:p,matched=soft_predict(model,z[:,cols],cols,names)
        finally:final_core.CONTINUOUS=previous
        saved=pd.read_csv(ROOT/e['prediction_path'],float_precision='round_trip')
        np.testing.assert_array_equal(saved.pid,te);np.testing.assert_array_equal(saved.y_true,frame.label_365d)
        np.testing.assert_allclose(p,saved.score,atol=1e-12,rtol=1e-12)
        error=float(np.max(np.abs(p-saved.score)));errors.append(error)
        modelrows.append(dict(task='nlst',stage=e['stage'],max_error=error,**metrics(frame.label_365d,p)))
    assert not development&confirmation
    nref=pd.read_csv(ROOT/'source_data/statistics/补实验/NLST最终患者级验证_20261006_120719/最终确认集指标.csv',float_precision='round_trip')
    for f in (ROOT/'predictions/nlst_confirmation_all').glob('*.csv'):
        model=f.name.removesuffix('_evaluation_预测.csv');g=pd.read_csv(f,float_precision='round_trip')
        compare(metrics(g.y_true,g.score),nref[nref.model==model].iloc[0])
    pd.DataFrame(modelrows).to_csv(out/'restored_model_metrics.csv',index=False)
    summary=dict(passed=True,original_baseline_conditions=len(rows),original_prediction_rows=len(pred),
        restored_original_models=10,restored_nlst_primary_models=11,nlst_confirmation_conditions=len(nref),
        maximum_prediction_error=max(errors),original_train_test_disjoint=True,
        nlst_development_confirmation_disjoint=True,full_training_rerun=False)
    (out/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

def retrain_original(stage,out):
    e=next(e for e in read(ROOT/'config/original_models.json') if f"repeat{e['repeat']}_fold{e['fold']}"==stage)
    data=pd.read_csv(ROOT/'data/original.csv',float_precision='round_trip')
    tr=np.asarray(e['training_rows'])-1;cols=np.asarray(e['columns'])
    model=fit_soft(data.iloc[tr,:-1].to_numpy(float)[:,cols],data.iloc[tr,-1].to_numpy(int),cols,
        list(data.columns[:-1]),e['seed'],e['lambda_value'],out)
    frozen=restore(ROOT/e['model_path'])
    for key in ['beta','r','w']:np.testing.assert_allclose(getattr(model,key),getattr(frozen,key),atol=1e-9,rtol=1e-9)
    (out/'retraining_verification.json').write_text(json.dumps(dict(passed=True,task='original',stage=stage,
        generations=200,features_and_lambda='frozen training-only selection',
        max_parameter_error=max(float(np.max(np.abs(getattr(model,k)-getattr(frozen,k)))) for k in ['beta','r','w'])),indent=2),encoding='utf-8')
    print('Conditional final fit matches the frozen model. SHAP and lambda choices were loaded from the frozen manifest.')

def retrain_nlst(stage,out):
    from nlst_core import supported_boundaries,rule_fit
    e=next(e for e in read(ROOT/'config/nlst_models.json') if e['stage']==stage)
    data=pd.read_csv(ROOT/'data/nlst.csv').set_index('pid').loc[e['training_pids']]
    X,names=transform(data,e['preprocessing']);cols=np.asarray(e['columns']);X=X[:,cols]
    y=data.label_365d.to_numpy(int);names=[names[c] for c in cols]
    settings=read(ROOT/e['model_path']/'训练信息.json')
    bounds=[supported_boundaries(X[:,j],y,settings['min_child']) for j in range(len(cols))]
    frozen=restore(ROOT/e['model_path'])
    for b,expected in zip(bounds,frozen.boundaries):np.testing.assert_array_equal(b,expected)
    out.mkdir(parents=True,exist_ok=True)
    model,info=rule_fit(X,y,names,e['seed'],True,.05,out,custom_boundaries=bounds,sigma0=settings['sigma0'])
    for key in ['beta','r','w']:np.testing.assert_allclose(getattr(model,key),getattr(frozen,key),atol=1e-9,rtol=1e-9)
    (out/'retraining_verification.json').write_text(json.dumps(dict(passed=True,task='nlst',stage=stage,
        generations=200,preprocessing_features_and_settings='frozen training-only selection',
        max_parameter_error=max(float(np.max(np.abs(getattr(model,k)-getattr(frozen,k)))) for k in ['beta','r','w'])),indent=2),encoding='utf-8')
    print('Conditional NLST final fit matches the frozen model. Preprocessing, features and inner-selected settings were loaded from the manifest.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'verification_output')
    parser.add_argument('--retrain-original',help='Optional final fit, e.g. repeat1_fold1; uses frozen features and selected lambda')
    parser.add_argument('--retrain-nlst',help='Optional final fit, e.g. confirmation; uses frozen preprocessing and selected settings')
    args=parser.parse_args()
    if args.retrain_original:retrain_original(args.retrain_original,args.output)
    elif args.retrain_nlst:retrain_nlst(args.retrain_nlst,args.output)
    else:run(args.output)
