"""Fixed SHAP1000/IBRB1000 pipeline: two repeated stratified five-fold evaluations."""
from pathlib import Path
import sys,json
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE/'依赖'));sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import lightgbm as lgb
import 论文配置引擎 as eng
from 重复五折与内层优化 import fold_ids,subset,shap_fit,save_rule
from 连续规则软匹配验证 import fit_soft
from 连续软匹配交付核验 import independent_soft
from 相邻规则平滑交付核验 import independent_penalty
from CMA代数延长小验证 import restore
from NLST训练内可行性评估 import fit_logistic,lr_predict,score,write_json
from scipy.stats import rankdata

def main():
    out=HERE/('原数据1000配置重复五折_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    (HERE/'最新原数据1000交叉验证目录.txt').write_text(str(out),encoding='utf-8')
    data=pd.read_csv(ROOT/'Lung Cancer Dataset.csv')
    X,y=data.iloc[:,:-1].to_numpy(float),data.iloc[:,-1].to_numpy(int)
    names=list(data.columns[:-1])
    seeds=[20262041,20262042]
    write_json(out/'运行前固定方案.json',dict(outer_seeds=seeds,folds=5,repeats=2,
        outer_training_pool=4000,SHAP_n=1000,IBRB_n=1000,outer_test_n=1000,
        SHAP='quantile alpha.5 standardized X,y;200 rounds;top7 canonical order',
        methods=['registered hard','soft fixed lambda.05'],
        rules='positive IG max20,parent5,Laplace,r.9,w1/L,CMA200sigma.5',
        baseline='LR fixed ridge1;binary LightGBM200 lr.05 leaves31 depth6 minleaf20 L2=1; same7/all17',
        selection='none; settings fixed before evaluation; one prespecified optimizer seed per fold',
        scope='development repeated CV on previously investigated original benchmark; all5000 evaluated once per repeat',
        lock='all models fitted before outer scores; no automatic adoption',NLST_test_scored_n=0))
    cases=[];members=[];entries=[]
    for repeat,outer_seed in enumerate(seeds,1):
        ids=fold_ids(y,5,outer_seed)
        for f in range(5):
            tr,te=np.flatnonzero(ids!=f),np.flatnonzero(ids==f)
            seed=outer_seed+1000+f
            used=subset(tr,y,1000,seed)
            assert len(tr)==4000 and len(te)==1000 and not np.intersect1d(used,te).size
            folder=out/f'repeat{repeat}_fold{f+1}';folder.mkdir()
            cases.append((repeat,f+1,seed,used,te,folder))
            for role,rows in [('outer_train',tr),('model_train',used),('test',te)]:
                members.extend(dict(repeat=repeat,fold=f+1,subset=role,source_row_1based=int(r+1)) for r in rows)
    pd.DataFrame(members).to_csv(out/'预先固定成员.csv',index=False)
    for repeat,fold,seed,tr,te,folder in cases:
        cols=shap_fit(X[tr],y[tr],seed,folder/'SHAP')
        def entry(model,family,columns,path):
            entries.append(dict(repeat=repeat,fold=fold,model=model,family=family,seed=seed,
                columns=columns.tolist(),training_rows=(tr+1).tolist(),test_rows=(te+1).tolist(),model_path=str(path)))
        bounds=[eng.supervised_boundaries(X[tr,c],y[tr]) for c in cols]
        hard=eng.build_ibrb(X[tr][:,cols],y[tr],seed=seed,custom_boundaries=bounds)
        save_rule(folder/'hard',hard,cols,dict(seed=seed,generations=200,sigma0=.5,K_max=20,n_min=5))
        entry('IBRB_hard1000','hard',cols,folder/'hard')
        fit_soft(X[tr][:,cols],y[tr],cols,names,seed,.05,folder/'soft',custom_boundaries=bounds)
        entry('IBRB_soft1000','soft',cols,folder/'soft')
        for suffix,c in [('same7',cols),('all17',np.arange(len(names)))]:
            lr=fit_logistic(X[tr][:,c],y[tr])
            path=folder/f'LR_{suffix}.npz'
            np.savez(path,weights=lr[0],mean=lr[1],sd=lr[2])
            entry('LR1000_'+suffix,'LR',c,path)
            gb=lgb.train(dict(objective='binary',learning_rate=.05,max_depth=6,num_leaves=31,
                min_data_in_leaf=20,lambda_l2=1.,seed=seed,num_threads=2,verbosity=-1,
                deterministic=True,force_col_wise=True),lgb.Dataset(X[tr][:,c],label=y[tr]),num_boost_round=200)
            path=folder/f'LGBM_{suffix}.txt';path.write_text(gb.model_to_string(),encoding='utf-8')
            entry('LGBM1000_'+suffix,'LGBM',c,path)
        write_json(out/'模型清单.json',entries)
        print(f'FITTED repeat{repeat} fold{fold} rules{hard.L}',flush=True)
    write_json(out/'全部评价前锁定.json',dict(all_models_saved=True,rule_models=20,baseline_models=40))
    records=[];predictions=[];checked=0
    for e in entries:
        c=np.array(e['columns']);tr=np.array(e['training_rows'])-1;path=Path(e['model_path'])
        if e['family'] in ['hard','soft']:
            model=restore(path)
            initial=eng.build_ibrb(X[tr][:,c],y[tr],optimize=False)
            for a,b in zip(model.boundaries,initial.boundaries): np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(np.load(path/'模型.npz')['initial_parameters'],initial.initial_parameters_)
        elif e['family']=='LR':
            saved=np.load(path);model=(saved['weights'],saved['mean'],saved['sd'])
        else: model=lgb.Booster(model_str=path.read_text(encoding='utf-8'))
        for role,field in [('train','training_rows'),('holdout','test_rows')]:
            idx=np.array(e[field])-1;values=X[idx][:,c]
            if e['family']=='soft': p,matched=independent_soft(model,values,c,names)
            elif e['family']=='hard': p,matched=eng.predict_proba(model,values)
            elif e['family']=='LR': p=lr_predict(model,values);matched=np.ones(len(idx),bool)
            else: p=model.predict(values);matched=np.ones(len(idx),bool)
            frame=pd.DataFrame(dict(source_row_1based=idx+1,y_true=y[idx],score=p,matched=matched))
            pred_path=out/f'repeat{e["repeat"]}_fold{e["fold"]}'/f'{e["model"]}_{role}_预测.csv'
            frame.to_csv(pred_path,index=False)
            read=pd.read_csv(pred_path,float_precision='round_trip')
            np.testing.assert_allclose(read.score,p,atol=1e-12,rtol=0)
            metric=score(y[idx],p)
            pos=y[idx].sum();neg=len(idx)-pos
            auc=(rankdata(read.score)[y[idx]==1].sum()-pos*(pos+1)/2)/(pos*neg)
            np.testing.assert_allclose([auc,np.mean((read.score-y[idx])**2)],[metric['auc'],metric['mse']],atol=1e-12)
            if role=='train' and e['family'] in ['hard','soft']:
                cfg=json.loads((path/'配置.json').read_text(encoding='utf-8'))
                np.testing.assert_allclose(cfg['fit_info']['training_mse'],metric['mse'],atol=1e-12)
                if e['family']=='soft':
                    _,penalty=independent_penalty(model,c,names)
                    np.testing.assert_allclose(cfg['fit_info']['regularized_objective'],metric['mse']+.05*penalty,atol=1e-12)
            records.append(dict(repeat=e['repeat'],fold=e['fold'],model=e['model'],subset=role,
                rules=model.L if e['family'] in ['hard','soft'] else 0,unmatched_n=int((~matched).sum()),**metric))
            if role=='holdout': predictions.append(frame.assign(repeat=e['repeat'],fold=e['fold'],model=e['model']))
            checked+=1
    result=pd.DataFrame(records);result.to_csv(out/'全部指标.csv',index=False)
    hold=result[result.subset.eq('holdout')]
    hold.groupby('model')[['accuracy','auc','mse','average_precision']].agg(['mean','std']).to_csv(out/'十折指标汇总.csv')
    pooled=pd.concat(predictions,ignore_index=True);pooled.to_csv(out/'全部外层预测.csv',index=False)
    per_repeat=[]
    for (repeat,model),part in pooled.groupby(['repeat','model']):
        assert len(part)==5000 and part.source_row_1based.nunique()==5000
        per_repeat.append(dict(repeat=repeat,model=model,**score(part.y_true.to_numpy(),part.score.to_numpy())))
    pd.DataFrame(per_repeat).to_csv(out/'每次重复汇总.csv',index=False)
    write_json(out/'独立交付核验.json',dict(passed=True,rule_models=20,baseline_models=40,prediction_files=checked,
        independent_soft_interpolation=True,independent_penalty=True,rank_auc_checked=True,
        all_records_evaluated_once_per_repeat=True,NLST_test_scored_n=0))
    write_json(out/'完成记录.json',dict(completed=True,outer_folds=10,SHAP_fits=10,rule_fits=20,baseline_fits=40,
        automatic_adoption=False,NLST_test_scored_n=0))
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__': main()
