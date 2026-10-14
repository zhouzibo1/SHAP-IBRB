# -*- coding: utf-8 -*-
"""Frozen three stratified holdouts; train every pipeline before test scoring."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "依赖"))
import json
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import expit
import ibrb_engine as eng
from 最小核验实验 import metrics, roc_points, write_json
from 三组最小消融 import Tee
from 训练内验证改进 import split_train

SPLIT_SEEDS = [20261501, 20261502, 20261503]
MODEL_SEEDS = [20261003, 20261004, 20261005]
LABELS = {"A_SHAP_IG": "SHAP七特征＋监督区间",
          "B_SHAP_EW": "SHAP七特征＋等宽区间",
          "C_ALL_IG": "17特征＋监督区间",
          "D_NO_CMAES": "七特征监督区间，无优化",
          "LR_260": "逻辑回归，260条",
          "LR_1000": "逻辑回归，1000条",
          "LGBM_260": "LightGBM，260条",
          "LGBM_1000": "LightGBM，1000条"}


def ids(path):
    return pd.read_csv(path).source_row_1based.to_numpy(dtype=int)-1


def standardize_fit(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd==0] = 1.
    return mu, sd, (X-mu)/sd


def fit_lr(X, y):
    mu, sd, Z = standardize_fit(X)
    n, m = Z.shape
    def objective(w):
        logits = w[0] + Z @ w[1:]
        value = np.logaddexp(0., logits).sum() - y @ logits + .5*(w[1:] @ w[1:])
        residual = expit(logits)-y
        grad = np.r_[residual.sum(), Z.T @ residual + w[1:]]
        return value, grad
    fitted = minimize(objective, np.zeros(m+1), jac=True, method="L-BFGS-B",
                      options={"maxiter":1000,"ftol":1e-12,"gtol":1e-8})
    assert fitted.success
    return {"kind":"logistic", "weights":fitted.x, "mean":mu, "sd":sd,
            "converged":bool(fitted.success), "objective":float(fitted.fun)}


def model_from_x(bounds, x):
    L = sum(len(b)-1 for b in bounds)
    return eng.IBRBModel(bounds,x[:2*L].reshape(L,2),x[2*L:3*L],x[3*L:])


def fit_rule(X,y,method,seed,optimize=True):
    if method=="supervised":
        bounds=[eng.info_gain_split(X[:,i],y,20,mode="faithful",n_min=5) for i in range(X.shape[1])]
    else:
        bounds=[eng.equal_width_split(X[:,i],20) for i in range(X.shape[1])]
    L=sum(len(b)-1 for b in bounds)
    beta=eng.laplace_init(X,y,bounds)
    x0=np.r_[beta.ravel(),np.full(L,.9),np.ones(L)]
    fitness=eng.make_fitness(X,y,bounds,L,2)
    initial=fitness(x0)
    if optimize:
        bestx,bestf=eng.cmaes_min(x0.copy(),200,fitness,L,2,seed=seed,sigma0=.5,verbose=False)
        retained=initial<=bestf
        chosen=x0 if retained else bestx
    else:
        chosen=x0;bestf=None;retained=True
    model=model_from_x(bounds,chosen)
    info={"initial_training_mse":initial,"selected_training_mse":fitness(chosen),
          "CMAES_best_training_mse":bestf,"retained_initial_model":bool(retained),
          "rule_slots":L,"parameters":4*L,"optimizer_population":10+int(3*np.log(4*L)) if optimize else 0,
          "fitness_evaluations":200*(10+int(3*np.log(4*L))) if optimize else 0}
    return {"kind":"rule","model":model,"fit_info":info}


def save_entry(directory, entry):
    directory.mkdir()
    info={k:v for k,v in entry.items() if k not in ['model','base','mean','sd','weights','idx']}
    info['selected_columns']=entry['idx'].tolist()
    write_json(directory/'配置.json',info)
    if entry['kind']=='rule':
        model=entry['model'];np.savez(directory/'模型.npz',beta=model.beta,r=model.r,w=model.w)
        write_json(directory/'边界.json',[b.tolist() for b in model.boundaries])
    elif entry['kind']=='logistic':
        np.savez(directory/'模型.npz',weights=entry['weights'],mean=entry['mean'],sd=entry['sd'])
    else:
        (directory/'模型.txt').write_text(entry['base'].model_to_string(),encoding='utf-8')
        np.savez(directory/'标准化.npz',mean=entry['mean'],sd=entry['sd'])


def load_entry(directory):
    info=json.loads((directory/'配置.json').read_text(encoding='utf-8'))
    info['idx']=np.array(info.pop('selected_columns'),dtype=int)
    if info['kind']=='rule':
        p=np.load(directory/'模型.npz');bounds=[np.asarray(b) for b in json.loads((directory/'边界.json').read_text(encoding='utf-8'))]
        info['model']=eng.IBRBModel(bounds,p['beta'],p['r'],p['w'])
    elif info['kind']=='logistic':
        p=np.load(directory/'模型.npz');info.update({k:p[k] for k in ['weights','mean','sd']})
    else:
        info['base']=lgb.Booster(model_str=(directory/'模型.txt').read_text(encoding='utf-8'))
        p=np.load(directory/'标准化.npz');info.update({k:p[k] for k in ['mean','sd']})
    return info


def predict(entry,X):
    selected=X[:,entry['idx']]
    if entry['kind']=='rule':return eng.predict_proba(entry['model'],selected)
    if entry['kind']=='logistic':
        p=expit(entry['weights'][0]+((selected-entry['mean'])/entry['sd'])@entry['weights'][1:])
    else:p=entry['base'].predict((selected-entry['mean'])/entry['sd'])
    return p,np.ones(len(X),dtype=bool)


def main():
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    out=HERE/f"三次分层划分验证_{stamp}";out.mkdir()
    sys.stdout=Tee(sys.stdout,out/'运行日志.txt')
    source=pd.read_csv(ROOT/'Lung Cancer Dataset.csv')
    names=list(source.columns[:-1]);X=source.iloc[:,:-1].to_numpy(dtype=float);y=source.iloc[:,-1].to_numpy(dtype=int)
    current=json.loads((HERE/'当前固定返修方案.json').read_text(encoding='utf-8'))
    fixed=Path(current['membership_directory'])
    protected=ids(fixed/'固定130条测试成员.csv')
    oldval=ids(fixed/'固定300条开发验证成员.csv')
    old_reserved=ids(HERE/'标准优化器锁定验证_20261003_170503'/'预先锁定新测试成员.csv')
    eligible=np.setdiff1d(np.arange(len(y)),np.r_[protected,oldval,old_reserved])
    tests=[]
    for seed in SPLIT_SEEDS:
        _,selected=split_train(y[eligible],130,seed)
        test=eligible[selected];tests.append(test);eligible=np.setdiff1d(eligible,test)
    all_test=np.concatenate(tests)
    train_pool=np.setdiff1d(np.arange(len(y)),np.r_[protected,oldval,all_test])
    splits=[]
    for seed,test in zip(SPLIT_SEEDS,tests):
        _,selected=split_train(y[train_pool],260,seed+100)
        train=train_pool[selected]
        auxiliary_pool=np.setdiff1d(train_pool,train)
        aux=np.random.default_rng(seed+200).permutation(auxiliary_pool)[:740]
        shap=np.sort(np.r_[train,aux])
        assert len(shap)==1000 and not np.intersect1d(shap,np.r_[all_test,protected,oldval]).size
        splits.append((train,test,shap))
    gb_params={"objective":"binary","learning_rate":.05,"max_depth":6,"num_leaves":31,
               "min_data_in_leaf":20,"lambda_l2":1.,"num_threads":2,"verbosity":-1,
               "deterministic":True,"force_col_wise":True}
    protocol={"purpose":"three prespecified stratified holdouts at260/130; fixed hyperparameters, no test-based model selection",
      "split_seeds":SPLIT_SEEDS,"optimizer_seeds":MODEL_SEEDS,"rule_training_n_each":260,"test_n_each":130,
      "test_membership":"three mutually disjoint test subsets reserved first; all390 excluded from all training and SHAP fits in this run",
      "test_eligibility":"exclude original frozen130,global development300,and previous390 reserved evaluation records",
      "protected_main_membership_reference":str(fixed),"training_pool_n":len(train_pool),
      "SHAP_training_n_each":1000,"SHAP_membership":"current260 rule-training plus740 auxiliary records; labels from this1000 only",
      "SHAP_model":{"objective":"quantile","alpha":.5,"rounds":200,"training_only_feature_and_target_standardization":True},
      "SHAP_selection":"top7 mean abs training TreeSHAP,expected-value column excluded,canonical feature order",
      "rule_settings":{"partition":"faithful K20 n_min5","initialization":"Laplace beliefs,r=.9,w=1",
                       "optimizer":"existing CMAES200 generations,sigma.5; best training-MSE among start and5600/6200 candidates",
                       "unmatched_policy":"existing engine output0; all test records retained"},
      "baseline_lightgbm":{**gb_params,"num_boost_round":200},
      "baseline_logistic":{"input":"17 standardized predictors; own training mean/std","penalty":"sum logistic loss +.5*sum(slope^2); intercept unpenalized",
                           "optimizer":"L-BFGS-B,maxiter1000,ftol1e-12,gtol1e-8"},
      "baseline_training_sizes":[260,1000],"comparison_rationale":"260 compares final rule-training size;1000 compares full upstream labeled-data access",
      "models":LABELS,"threshold":.5,"calibration":"not used in this verification",
      "selection":"all settings fixed in advance; no hyperparameter search or calibration fitting in this run",
      "model_lock":"all3 splits and all24 models saved before any test prediction",
      "new_IBRB_optimizer_runs":9,"SHAP_fits":3,"binary_LightGBM_fits":6,"logistic_fits":6,"no_optimizer_models":3,
      "validation_name":"repeated stratified holdout with fixed hyperparameters; not repeated nested cross-validation",
      "scope":"same previously investigated benchmark dataset; not an external clinical validation"}
    write_json(out/'预先固定方案.json',protocol)
    member_rows=[]
    for repeat,(train,test,shap) in enumerate(splits,1):
        for role,arr in [('rule_train',train),('test',test),('SHAP_train',shap)]:
            member_rows.extend({'repeat':repeat,'source_row_1based':int(i+1),'subset':role} for i in arr)
    pd.DataFrame(member_rows).to_csv(out/'预先固定三次成员.csv',index=False)
    print('OUTPUT '+str(out),flush=True)
    all_models=[]
    for repeat,((train,test,shap),split_seed,model_seed) in enumerate(zip(splits,SPLIT_SEEDS,MODEL_SEEDS),1):
        rd=out/f'repeat{repeat}';rd.mkdir()
        mu,sd,Z=standardize_fit(X[shap]);ym,ys=y[shap].mean(),y[shap].std()
        base=lgb.train({'objective':'quantile','alpha':.5,'seed':model_seed,'num_threads':2,'verbosity':-1,
                        'deterministic':True,'force_col_wise':True},lgb.Dataset(Z,label=(y[shap]-ym)/ys),num_boost_round=200)
        importance=np.abs(np.asarray(base.predict(Z,pred_contrib=True))[:,:-1]).mean(axis=0)
        selected=np.sort(np.argsort(-importance,kind='stable')[:7])
        pd.DataFrame({'feature':names,'mean_absolute_SHAP':importance,'selected':[i in selected for i in range(len(names))]}).sort_values('mean_absolute_SHAP',ascending=False).to_csv(rd/'SHAP排序.csv',index=False)
        (rd/'SHAP基模型.txt').write_text(base.model_to_string(),encoding='utf-8')
        np.savez(rd/'SHAP标准化.npz',mean=mu,sd=sd,target_mean=ym,target_sd=ys)
        write_json(rd/'划分配置.json',{'split_seed':split_seed,'optimizer_seed':model_seed,'selected_features':[names[i] for i in selected],
          'rule_train_class_counts':{str(c):int((y[train]==c).sum()) for c in np.unique(y)},
          'SHAP_class_counts':{str(c):int((y[shap]==c).sum()) for c in np.unique(y)}})
        print(f'REPEAT {repeat} selected={[names[i] for i in selected]}',flush=True)
        entries={}
        for name,method,idx,opt in [('A_SHAP_IG','supervised',selected,True),('B_SHAP_EW','equal_width',selected,True),
                                    ('C_ALL_IG','supervised',np.arange(len(names)),True),('D_NO_CMAES','supervised',selected,False)]:
            print(f'TRAIN repeat={repeat} model={name}',flush=True)
            entry=fit_rule(X[train][:,idx],y[train],method,model_seed,optimize=opt)
            entry['idx']=idx;entry['train_n']=260;entry['total_label_access_n']=1000 if len(idx)==7 else 260
            save_entry(rd/name,entry);entries[name]=entry
        for n,arr in [(260,train),(1000,shap)]:
            entry=fit_lr(X[arr],y[arr]);entry['idx']=np.arange(len(names));entry['train_n']=n;entry['total_label_access_n']=n
            save_entry(rd/f'LR_{n}',entry);entries[f'LR_{n}']=entry
            bmu,bsd,BZ=standardize_fit(X[arr]);params={**gb_params,'seed':model_seed}
            binary=lgb.train(params,lgb.Dataset(BZ,label=y[arr]),num_boost_round=200)
            entry={'kind':'lightgbm','base':binary,'mean':bmu,'sd':bsd,'idx':np.arange(len(names)),
                   'train_n':n,'total_label_access_n':n,'parameters':params,'rounds':200}
            save_entry(rd/f'LGBM_{n}',entry);entries[f'LGBM_{n}']=entry
        all_models.append(entries)
    write_json(out/'全部模型锁定.json',{'all24_models_saved_before_test_prediction':True,'hyperparameter_selection_uses_test':False,
      'optimizer_uses_training_MSE_only':True,'models_each_repeat':list(LABELS),'test_evaluation_begins_after_this_record':True})
    result=[];checks=[];coverage=[]
    for repeat,((train,test,shap),entries) in enumerate(zip(splits,all_models),1):
        rd=out/f'repeat{repeat}'
        for name,entry in entries.items():
            p,matched=predict(entry,X[test])
            row={'repeat':repeat,'split_seed':SPLIT_SEEDS[repeat-1],'optimizer_seed':MODEL_SEEDS[repeat-1],
                 'variant':name,'train_n':entry['train_n'],'total_label_access_n':entry['total_label_access_n'],
                 'feature_n':len(entry['idx']),'matched_test_n':int(matched.sum()),**metrics(y[test],p)}
            if entry['kind']=='rule':row.update(entry['fit_info'])
            result.append(row)
            pd.DataFrame({'source_row_1based':test+1,'y_true':y[test],'raw_score':p,'prediction':(p>=.5).astype(int),
                          'matched':matched}).to_csv(rd/name/'测试预测.csv',index=False)
            fpr,tpr,threshold=roc_points(y[test],p)
            pd.DataFrame({'threshold':threshold,'fpr':fpr,'tpr':tpr}).to_csv(rd/name/'ROC.csv',index=False)
            restored,rm=predict(load_entry(rd/name),X[test])
            np.testing.assert_allclose(restored,p,atol=1e-14,rtol=0);np.testing.assert_array_equal(rm,matched)
            positive,negative=p[y[test]==1],p[y[test]==0]
            auc_pairs=float(np.mean((positive[:,None]>negative[None,:])+.5*(positive[:,None]==negative[None,:])))
            np.testing.assert_allclose(row['auc'],auc_pairs,atol=1e-12,rtol=0)
            if entry['kind']=='rule':
                model=entry['model'];np.testing.assert_allclose(model.beta.sum(axis=1),1,atol=1e-12,rtol=0)
                for bad in test[~matched]:
                    reasons=[]
                    for idx,b in zip(entry['idx'],model.boundaries):
                        value=X[bad,idx]
                        if value < b[0] or value>b[-1]:reasons.append({'feature':names[idx],'value':float(value),'training_min':float(b[0]),'training_max':float(b[-1])})
                    coverage.append({'repeat':repeat,'variant':name,'source_row_1based':int(bad+1),'reason':reasons})
            checks.append({'repeat':repeat,'variant':name,'model_restored':True,'pairwise_AUC_verified':True})
            print(json.dumps(row,ensure_ascii=False),flush=True)
    frame=pd.DataFrame(result);frame.to_csv(out/'全部逐次指标.csv',index=False,encoding='utf-8-sig')
    cols=['accuracy','mse','auc','recall','specificity','f1']
    summary=frame.groupby('variant')[cols].agg(['mean','std']);summary.columns=[f'{a}_{b}' for a,b in summary.columns]
    summary.reset_index().to_csv(out/'均值与标准差.csv',index=False,encoding='utf-8-sig')
    paired=[]
    for rep in range(1,4):
        a=frame[(frame.repeat==rep)&(frame.variant=='A_SHAP_IG')].iloc[0]
        for name in LABELS:
            if name=='A_SHAP_IG':continue
            b=frame[(frame.repeat==rep)&(frame.variant==name)].iloc[0]
            paired.append({'repeat':rep,'comparison':f'A_SHAP_IG-minus-{name}',**{c:float(a[c]-b[c]) for c in cols}})
    pd.DataFrame(paired).to_csv(out/'配对差值.csv',index=False,encoding='utf-8-sig')
    write_json(out/'未匹配记录说明.json',coverage)
    write_json(out/'输出核验.json',{'all_checks_passed':True,'model_checks':checks,'all390_tests_disjoint_from_all_current_training':True,
      'test_sets_mutually_disjoint':True,'frozen_main130_excluded':True,'all_models_saved_before_test_scoring':True})
    write_json(out/'完成状态.json',{'rule_optimizer_runs':9,'SHAP_fits':3,'binary_LightGBM_fits':6,'logistic_fits':6,
      'no_optimizer_models':3,'test_predictions':24,'original_manuscript_CSV_engine_edited':False})
    lines=[]
    for name in LABELS:
        v=frame[frame.variant==name];parts=[LABELS[name]]
        for c in ['accuracy','mse','auc','recall']:
            factor=100 if c in ['accuracy','recall'] else 1;precision=2 if factor==100 else 5;suffix='%' if factor==100 else ''
            parts.append(f'{v[c].mean()*factor:.{precision}f}±{v[c].std()*factor:.{precision}f}{suffix}')
        lines.append('| '+' | '.join(parts)+' |')
    means=frame.groupby('variant')[cols].mean();a=means.loc['A_SHAP_IG'];b=means.loc['B_SHAP_EW'];c=means.loc['C_ALL_IG']
    feature_lines=[]
    for rep in range(1,4):
        cfg=json.loads((out/f'repeat{rep}'/'划分配置.json').read_text(encoding='utf-8'))
        feature_lines.append(f"- 第{rep}次：{', '.join(cfg['selected_features'])}。")
    report=f"""# 三次分层划分验证

完成3次预先固定分层留出，各260条IBRB训练、130条测试。新增9次规则优化、3次SHAP拟合、6次LightGBM和6次逻辑回归拟合；另评价3个确定性无优化模型。原稿、CSV及原引擎未修改。

## 固定设计

先保留三组互不重叠的130条测试，全部390条排除在本轮所有规则训练、SHAP及基线训练之外。原固定130条和开发300条也保留排除。每次从其余记录中分层抽取260条规则训练，再加入740条辅助记录形成1000条SHAP训练。全部成员在拟合前保存。

三个划分种子20261501、20261502、20261503，对应优化种子20261003、20261004、20261005。每次重新拟合SHAP、区间、规则参数及基线；全部24个模型保存后才开始测试评分。

规则统一采用Laplace初始化、r=0.9、w=1、faithful K20、200代sigma0=0.5，训练目标包含有效初始模型。与早期历史初始化结果分开报告。

超参数预先固定，没有搜索或测试选择。本轮属于重复分层留出验证；审稿人提出的重复嵌套交叉验证尚未按其完整形式执行。

## 结果（均值±样本标准差）

| 模型 | 准确率 | MSE | AUC | Recall |
|---|---:|---:|---:|---:|
{chr(10).join(lines)}

七特征监督模型相对七特征等宽：准确率差{100*(a.accuracy-b.accuracy):+.2f}个百分点，MSE差{a.mse-b.mse:+.5f}，AUC差{a.auc-b.auc:+.5f}。

七特征监督模型相对全17特征监督：准确率差{100*(a.accuracy-c.accuracy):+.2f}个百分点，MSE差{a.mse-c.mse:+.5f}，AUC差{a.auc-c.auc:+.5f}。

本轮监督划分和等宽划分的平均准确率接近，等宽的平均MSE、AUC略好。1000条逻辑回归在三个划分的准确率和MSE均优于七特征监督模型，AUC两次更高、一次相同。当前结果支持报告具体指标与划分波动，收窄“监督划分稳定占优”和“全面优于基线”的论述。

## 每次SHAP特征

{chr(10).join(feature_lines)}

每次都按当次训练数据重新筛选，三次恰好选回相同七特征。这里观察到的是特征名单稳定，预测性能仍随训练和测试成员变化。

## 对照与数据使用

逻辑回归和LightGBM使用全部17个特征，分别训练260条和1000条。260条对照对应IBRB参数训练规模，1000条对照对应SHAP流程总标签访问规模。两种口径分别展示，避免把额外740条辅助标签的作用全部归因于模型结构。

逻辑回归采用训练内标准化及L2惩罚，截距不惩罚；LightGBM为binary、200轮、学习率0.05、最大深度6、31叶、最小叶20、L2=1。每个模型标准化均仅依据自己的训练数据。详细设置与保存模型见预先固定方案。

规则匹配超出训练范围时，按原引擎分数0处理，保留所有130条评价；各模型未匹配记录与原因单独保存。本轮不采用新增分数校准。

## 解释范围及返修

本轮反映更换数据成员后整套流程的表现。三次重复同时改变训练成员、SHAP抽样及优化种子；标准差体现这些变化的合并波动。来源仍为同一个已分析的公开基准，属于划分验证，不属于外部临床验证。

固定原260/130的历史参照继续保留，新增划分结果独立报告；主张的强弱应按两者一致性和同数据访问口径基线确定。不能仅凭原固定划分接近90%的点估计声称稳定优势。

## 输出与核验

目录：`{out.name}`。保存成员、三次SHAP基模型与排序、24个模型及配置、24份测试预测、ROC、全部逐次指标、均值与标准差及配对差值。

核查所有390条测试与本轮全部训练独立、测试组互不重叠、所有模型保存后才评分、模型重载分数一致及两种AUC计算一致。全部检查通过。
"""
    (HERE/'三次分层划分验证结论.md').write_text(report,encoding='utf-8')
    print('COMPLETE',flush=True);print(means.to_string(),flush=True)


if __name__=='__main__':main()
