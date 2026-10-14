"""Development comparison of information-gain child support 5/10/20, fixed K20."""
from pathlib import Path
import sys
import json
from datetime import datetime, timezone, timedelta
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '依赖'))
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import NLST训练内可行性评估 as run

def supported_boundaries(values, labels, min_child, max_intervals=20, n_min_parent=5):
    """Original greedy positive IG, restricted to cuts supporting both children."""
    current = [float(values.min()), float(values.max())]
    entropy = run.engine.archived.calc_entropy
    while len(current) - 1 < max_intervals:
        best_gain, best_split, best_index = -np.inf, None, None
        for index in range(len(current) - 1):
            mask = (values >= current[index]) & (values <= current[index + 1])
            v, y = values[mask], labels[mask]
            if len(y) < n_min_parent:
                continue
            unique = np.unique(v)
            parent = entropy(y)
            for point in (unique[:-1] + unique[1:]) / 2:
                left = v <= point
                nl, nr = int(left.sum()), int((~left).sum())
                if min(nl, nr) < min_child:
                    continue
                gain = parent - (nl * entropy(y[left]) + nr * entropy(y[~left])) / len(y)
                if gain > best_gain:
                    best_gain, best_split, best_index = gain, point, index
        if best_split is None or best_gain <= 0:
            break
        current.insert(best_index + 1, float(best_split))
    return np.asarray(current)

def main():
    reference = HERE / 'NLST训练内可行性评估_20261005_110426'
    out = HERE / ('NLST最小子区间对照_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    cohort = HERE.parent / '数据来源核查' / 'NLST患者级候选_20261005'
    prep = json.loads((reference / '训练内预处理.json').read_text(encoding='utf-8'))
    names = json.loads((reference / '完成记录.json').read_text(encoding='utf-8'))['selected_features']
    ids = {role: pd.read_csv(reference / f'{role}_成员.csv').pid.tolist() for role in ['SHAP','IBRB','validation']}
    members = pd.read_csv(cohort / '候选划分成员表.csv').set_index('pid')
    assert not members.loc[ids['SHAP'] + ids['validation'], 'in_test'].any()
    assert not set(ids['SHAP']) & set(ids['validation'])
    source = pd.read_csv(cohort / '患者级建模候选.csv')
    source = source[source.pid.isin(set(ids['SHAP']) | set(ids['validation']))].set_index('pid')
    assert run.preprocess_fit(source.loc[ids['SHAP']]) == prep
    encoded, all_names = run.transform(source.loc[ids['IBRB']], prep)
    columns = [all_names.index(name) for name in names]
    X, y = encoded[:, columns], source.loc[ids['IBRB'], 'label_365d'].to_numpy(int)
    V = run.transform(source.loc[ids['validation']], prep)[0][:, columns]
    vy = source.loc[ids['validation'], 'label_365d'].to_numpy(int)
    reference_bounds = [np.asarray(b) for b in json.loads((reference / 'soft_lambda0.05_seed20261003' / '边界.json').read_text(encoding='utf-8'))]
    # min_child=1 reproduces the legacy unconstrained split choices.
    for column, boundary in enumerate(reference_bounds):
        np.testing.assert_array_equal(supported_boundaries(X[:,column], y, 1), boundary)
    run.write_json(out / '运行前固定方案.json', {
        'reference': str(reference), 'SHAP': 'frozen binary1000', 'selected_features': names,
        'IBRB_n': len(y), 'positive_n': int(y.sum()), 'validation_n': len(vy),
        'min_child_candidates': [5,10,20], 'max_intervals': 20, 'n_min_parent': 5,
        'soft_matching': True, 'lambda': .05, 'CMA_generations': 200, 'sigma0': .5,
        'seeds': run.SEEDS, 'fixed_test_scored': False,
        'purpose': 'bounded root control of sparse IG children; no threshold or probability calibration search',
        'adoption': 'none; report all candidate seed results'})
    old = pd.read_csv(reference / '全部训练内指标.csv')
    rows = old[old.model.isin(['soft_lambda0.05','LR260_same7','LR1000_same7','training_prevalence_constant'])].to_dict('records')
    support_rows = []
    for minimum in [5,10,20]:
        bounds = [supported_boundaries(X[:,j], y, minimum) for j in range(X.shape[1])]
        rule_ids, matched = run.engine.match_rules(X, bounds)
        assert matched.all()
        count = np.bincount(rule_ids.ravel(), minlength=sum(len(b)-1 for b in bounds))
        assert count.min() >= minimum
        support_rows.append({'min_child': minimum, 'rules': len(count), 'minimum_support': int(count.min()),
                             'support_below10': int((count < 10).sum())})
        for seed in run.SEEDS:
            name = f'soft_minchild{minimum}'
            md = out / f'{name}_seed{seed}'
            model, info = run.rule_fit(X, y, names, seed, True, .05, md, custom_boundaries=bounds)
            info['min_child'] = minimum
            run.write_json(md / '训练信息.json', info)
            saved = np.load(md / '模型.npz')
            reloaded = run.engine.IBRBModel(bounds, saved['beta'], saved['r'], saved['w'])
            for role, values, labels in [('IBRB',X,y),('validation',V,vy)]:
                p, matched = run.predict_plan(model, run.plan_for(values, bounds, names, True))
                replay, _ = run.predict_plan(reloaded, run.plan_for(values, bounds, names, True))
                np.testing.assert_allclose(replay, p, atol=1e-12, rtol=1e-12)
                path = out / f'{name}_seed{seed}_{role}_预测.csv'
                pd.DataFrame({'pid': ids[role], 'y_true': labels, 'score': p}).to_csv(path, index=False)
                metrics = run.score(labels,p)
                read = pd.read_csv(path)
                np.testing.assert_allclose(np.mean((read.score-labels)**2), metrics['mse'], atol=1e-12)
                pos, neg = labels.sum(), len(labels)-labels.sum()
                auc = (rankdata(read.score)[labels==1].sum()-pos*(pos+1)/2)/(pos*neg)
                np.testing.assert_allclose(auc, metrics['auc'], atol=1e-12)
                rows.append({'model':name,'seed':seed,'subset':role,'min_child':minimum,'rules':model.L,
                             'unmatched_n':int((~matched).sum()),'prediction_file':str(path),**metrics})
            pd.DataFrame(rows).to_csv(out / '全部比较指标.csv', index=False)
            print(f'READY min_child={minimum} seed={seed} rules={model.L} validation={metrics}', flush=True)
    result = pd.DataFrame(rows)
    result[result.subset.eq('validation')].groupby('model')[['auc','mse','accuracy','average_precision','recall']].agg(['mean','std']).to_csv(out / '开发验证汇总.csv')
    run.write_json(out / '规则支持度.json', support_rows)
    run.write_json(out / '交付核验.json', {'unconstrained_partition_reproduction': 'passed',
        'minimum_child_support': 'passed', 'saved_model_replay': 'passed', 'independent_MSE_AUC': 'passed',
        'new_model_n': 9, 'test_scored_n': 0, 'training_only_preprocessing': 'passed'})
    run.write_json(out / '完成记录.json', {'completed':True,'new_fits':9,'test_evaluated':False})
    (HERE / '最新NLST最小子区间目录.txt').write_text(str(out),encoding='utf-8')
    print('COMPLETE '+str(out),flush=True)

if __name__ == '__main__':
    main()
