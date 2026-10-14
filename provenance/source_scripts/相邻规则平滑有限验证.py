# -*- coding: utf-8 -*-
"""Smooth neighboring continuous-attribute rules, retaining joint CMA and ER."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(HERE/'依赖'))
sys.path.insert(0,str(ROOT))
import json
from datetime import datetime,timedelta,timezone
import numpy as np
import pandas as pd
import 论文配置引擎 as eng
from 最小核验实验 import metrics,write_json
from 重复五折与内层优化 import save_rule
from CMA代数延长小验证 import restore
from 子区间样本约束优化 import cases_from_source
from 三组最小消融 import Tee

GRID = [0.,.01,.05,.2]
CONTINUOUS = {'AGE','ENERGY_LEVEL','OXYGEN_SATURATION'}


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


def fit_smooth(X,y,columns,names,seed,strength,directory,reference=None,custom_boundaries=None):
    initial = eng.build_ibrb(X,y,max_intervals=20,n_min=5,optimize=False,custom_boundaries=custom_boundaries)
    L = initial.L
    pairs = neighbor_pairs(initial.boundaries,columns,names)
    raw_objective = eng.archived.make_fitness(X,y,initial.boundaries,L,2)
    if reference is not None:
        assert strength == 0
        model = restore(reference)
        for a,b in zip(initial.boundaries,model.boundaries):
            np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(initial.initial_parameters_,np.load(reference/'模型.npz')['initial_parameters'])
        old_path = reference/'训练信息.json'
        old = json.loads(old_path.read_text(encoding='utf-8')) if old_path.exists() else json.loads((reference/'配置.json').read_text(encoding='utf-8'))['fit_info']
        info = {k:v for k,v in old.items() if k != 'trace'}
        info['trace'] = [{('best_regularized_objective' if k == 'best_training_mse' else k):v
                         for k,v in row.items()} for row in old['trace']]
        info['regularized_objective'] = old['training_mse']
        info['initial_regularized_objective'] = old['initial_training_mse']
    else:
        def objective(x):
            return raw_objective(x)+strength*smoothness(x[:2*L].reshape(L,2),x[2*L:3*L],x[3*L:],pairs)
        chosen,info = eng.cmaes_min(initial.initial_parameters_,objective,L,seed,generations=200,sigma0=.5)
        model = eng.IBRBModel(initial.boundaries,chosen[:2*L].reshape(L,2),chosen[2*L:3*L],chosen[3*L:])
        info['regularized_objective'] = info.pop('training_mse')
        info['initial_regularized_objective'] = info.pop('initial_training_mse')
        info['trace'] = [{('best_regularized_objective' if k == 'best_training_mse' else k):v
                         for k,v in row.items()} for row in info['trace']]
    final_x = np.r_[model.beta.ravel(),model.r,model.w]
    info['training_mse'] = float(raw_objective(final_x))
    info['initial_training_mse'] = float(raw_objective(initial.initial_parameters_))
    info['smoothness_penalty'] = smoothness(model.beta,model.r,model.w,pairs)
    info['initial_smoothness_penalty'] = smoothness(initial.beta,initial.r,initial.w,pairs)
    info['penalty_contribution'] = strength*info['smoothness_penalty']
    info['objective_name'] = 'training_MSE + lambda*mean_squared_neighbor_beta_r_w_differences'
    np.testing.assert_allclose(info['regularized_objective'],info['training_mse']+info['penalty_contribution'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(info['initial_regularized_objective'],info['initial_training_mse']+strength*info['initial_smoothness_penalty'],atol=1e-12,rtol=0)
    model.initial_parameters_,model.fit_info_ = initial.initial_parameters_,info
    save_rule(directory,model,columns,{'lambda':strength,'continuous_features':sorted(CONTINUOUS),
        'neighbor_edge_count':len(pairs),'penalty_normalization':'mean over4E parameter differences',
        'K_max':20,'n_min':5,'generations':200,'sigma0':.5,'population_multiplier':1,'seed':seed,
        'reused_source':str(reference) if reference is not None else None,
        'boundary_source':'custom supplied training boundaries' if custom_boundaries is not None else 'supervised information gain'})
    np.save(directory/'相邻规则对.npy',pairs)
    return model


def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    out = HERE/f'相邻规则平滑有限验证_{stamp}'
    out.mkdir()
    (HERE/'最新相邻规则平滑验证目录.txt').write_text(str(out),encoding='utf-8')
    sys.stdout = Tee(sys.stdout,out/'运行日志.txt')
    baseline = Path((HERE/'最新260条同成员基线目录.txt').read_text(encoding='utf-8'))
    diagnostic = Path((HERE/'最新逻辑回归诊断目录.txt').read_text(encoding='utf-8'))
    prior_ref = Path((HERE/'最新置信度正则化验证目录.txt').read_text(encoding='utf-8'))
    write_json(out/'运行前固定方案.json',{'grid_lambda':GRID,'continuous_features':sorted(CONTINUOUS),
        'objective':'J=training_MSE+lambda/(4E)*sum_continuous_neighbor_edges(||theta_left-theta_right||^2),theta=(beta0,beta1,r,w)',
        'held_fixed':'SHAP7,1000/260/130,allcandidateIG,Kmax20,parent_n_min5,Laplace,r0.9,w1/L,joint beta_r_w,CMA200,sigma0.5,originalpopulation,identityinitialcovariance,projection,ER',
        'selection':'three training-internal folds mean raw validation MSE,ties follow grid order',
        'scope':'fixed130 plus all10 existing repeated5fold cases;independent selection in each case',
        'lock':'all11 choices and13 selected models saved before new outer evaluation',
        'interpretation':'finite exploratory method extension on previously investigated benchmark',
        'baseline260_reference':str(baseline),'LR7_reference':str(diagnostic),'Laplace_regularization_reference':str(prior_ref),
        'adoption':'requires explicit smoothness objective and train-only lambda choice;no automatic manuscript replacement'})
    source = pd.read_csv(ROOT/'Lung Cancer Dataset.csv')
    X,y = source.iloc[:,:-1].to_numpy(float),source.iloc[:,-1].to_numpy(int)
    names = list(source.columns[:-1])
    cases = cases_from_source(source)
    fitted,inner_rows = [],[]
    print('OUTPUT '+str(out),flush=True)
    for case in cases:
        name = case['case']
        assert len(case['train']) == 260 and len(case['shap']) == 1000
        assert np.isin(case['train'],case['shap']).all() and not np.intersect1d(case['shap'],case['test']).size
        if case['development'] is not None:
            assert not np.intersect1d(case['shap'],case['development']).size
        scores = []
        for fold in range(1,4):
            rd = case['inner_reference']/f'inner{fold}'
            members = pd.read_csv(rd/'成员.csv')
            train = pd.read_csv(rd/'规则训练成员.csv').source_row_1based.to_numpy(int)-1
            va = members[members.subset == 'validation'].source_row_1based.to_numpy(int)-1
            sh = members[members.subset == 'SHAP_train'].source_row_1based.to_numpy(int)-1
            assert np.isin(train,sh).all() and not np.intersect1d(sh,va).size
            assert np.isin(np.r_[sh,va],case['shap']).all()
            cols = np.array(json.loads((rd/'排序.json').read_text(encoding='utf-8'))['columns'])
            for strength in GRID:
                md = out/name/f'inner{fold}'/f'lambda{strength}'
                ref = rd/'K20_sigma0.5' if strength == 0 else None
                model = fit_smooth(X[train][:,cols],y[train],cols,names,case['inner_seed']+500+fold-1,strength,md,ref)
                p,matched = eng.predict_proba(model,X[va][:,cols])
                path = md/'内层预测.csv'
                pd.DataFrame({'source_row_1based':va+1,'y_true':y[va],'raw_score':p,'matched':matched}).to_csv(path,index=False)
                row = {'case':name,'inner_fold':fold,'lambda':strength,'rules':model.L,
                    'training_mse':model.fit_info_['training_mse'],'smoothness_penalty':model.fit_info_['smoothness_penalty'],
                    'regularized_objective':model.fit_info_['regularized_objective'],'reused':ref is not None,
                    'prediction_file':str(path),**metrics(y[va],p)}
                scores.append(row)
                inner_rows.append(row)
                print(f'INNER {name} f{fold} lambda={strength} rawMSE={row["mse"]:.5f}',flush=True)
        means = pd.DataFrame(scores).groupby('lambda').mse.mean()
        selected = min(GRID,key=lambda s:means.loc[s])
        write_json(out/name/'内层选择.json',{'selected_lambda':selected,'inner_mean_raw_mse':{float(k):float(v) for k,v in means.items()},
                                           'inner_reference':str(case['inner_reference'])})
        for seed in case['seeds']:
            ref = case['reference']/(f'A_full_seed{seed}' if case['repeat'] == 0 else 'A_full')
            md = out/name/f'选中模型_seed{seed}'
            model = fit_smooth(X[case['train']][:,case['columns']],y[case['train']],case['columns'],names,seed,selected,md,ref if selected == 0 else None)
            fitted.append((case,seed,selected,model,md,ref))
            print(f'FINAL {name} seed={seed} selected_lambda={selected}',flush=True)
        pd.DataFrame(inner_rows).to_csv(out/'全部内层指标.csv',index=False)
    write_json(out/'全部评价前锁定.json',{'all11_choices_and13_models_saved':True,'selection_metric':'raw inner validation MSE'})
    rows = []
    for case,seed,strength,chosen,md,ref in fitted:
        for label,model,model_dir,setting in [('IBRB_smooth_selected',chosen,md,strength),('IBRB_original',restore(ref),ref,0.)]:
            p_train,matched_train = eng.predict_proba(model,X[case['train']][:,case['columns']])
            assert matched_train.all()
            train_mse = float(np.mean((p_train-y[case['train']])**2))
            pairs = neighbor_pairs(model.boundaries,case['columns'],names)
            roughness = smoothness(model.beta,model.r,model.w,pairs)
            for subset,ids in [('test',case['test'])]+([('development',case['development'])] if case['development'] is not None else []):
                p,matched = eng.predict_proba(model,X[ids][:,case['columns']])
                path = out/case['case']/f'{label}_seed{seed}_{subset}_预测.csv'
                pd.DataFrame({'source_row_1based':ids+1,'y_true':y[ids],'raw_score':p,'matched':matched}).to_csv(path,index=False)
                values = metrics(y[ids],p)
                rows.append({'case':case['case'],'repeat':case['repeat'],'fold':case['fold'],
                    'scope':'fixed130' if case['repeat'] == 0 else 'repeated5fold','seed':seed,'model':label,
                    'subset':subset,'lambda':setting,'rules':model.L,'training_mse':train_mse,
                    'smoothness_penalty':roughness,'test_minus_train_mse':values['mse']-train_mse,
                    'model_directory':str(model_dir),'columns':json.dumps(case['columns'].tolist()),
                    'prediction_file':str(path),**values})
    for case in cases:
        controls = [('LR260_17',baseline/case['case']/'LR_260_test_预测.csv')]
        if case['repeat']:
            controls.append(('LR260_SHAP7',diagnostic/case['case']/'LR7_test1000_预测.csv'))
        for label,path in controls:
            p = pd.read_csv(path,float_precision='round_trip')
            np.testing.assert_array_equal(p.source_row_1based,case['test']+1)
            np.testing.assert_array_equal(p.y_true,y[case['test']])
            rows.append({'case':case['case'],'repeat':case['repeat'],'fold':case['fold'],
                'scope':'fixed130' if case['repeat'] == 0 else 'repeated5fold','model':label,'subset':'test',
                'prediction_file':str(path),**metrics(y[case['test']],p.raw_score.to_numpy(float))})
    previous = pd.read_csv(prior_ref/'全部评价指标.csv')
    for _,row in previous[previous.model == 'IBRB_regularized_selected'].iterrows():
        record = row.to_dict()
        record['model'] = 'IBRB_Laplace_regularized'
        rows.append(record)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'全部评价指标.csv',index=False)
    frame.groupby(['scope','subset','model'])[['accuracy','mse','auc','rules','training_mse','test_minus_train_mse','smoothness_penalty']].agg(['mean','std']).to_csv(out/'均值与标准差.csv')
    write_json(out/'完成记录.json',{'completed':True,'inner_candidates':len(inner_rows),'selected_models':len(fitted),
        'metric_rows':len(frame),'original_manuscript_modified':False})
    print(frame.groupby(['scope','subset','model'])[['accuracy','mse','auc','rules','training_mse','test_minus_train_mse','smoothness_penalty']].mean().to_string(),flush=True)
    print('COMPLETE '+str(out),flush=True)


if __name__ == '__main__':
    main()
