"""Fixed-budget NLST development evaluation. No fixed-test feature or score evaluation."""
from pathlib import Path
import sys
import json
import time
import argparse
from datetime import datetime, timezone, timedelta
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / '依赖'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit
from scipy.optimize import minimize
import 论文配置引擎 as engine
from 最小核验实验 import metrics, write_json

CONTINUOUS = ['age', 'baseline_nodule_count', 'index_long_diameter_mm', 'index_perpendicular_diameter_mm']
BINARY = ['male', 'cigsmok']
NOMINAL = ['race', 'index_lobe', 'index_margin', 'index_attenuation']
SEEDS = [20261003, 20261004, 20261005]

def preprocess_fit(frame):
    impute = {}
    categories = {}
    for column in CONTINUOUS:
        impute[column] = float(frame[column].median())
    for column in BINARY + NOMINAL:
        impute[column] = float(frame[column].mode().iloc[0])
    for column in NOMINAL:
        categories[column] = sorted(frame[column].fillna(impute[column]).unique().tolist())
    return {'impute': impute, 'categories': categories, 'fit_n': len(frame)}

def transform(frame, prep):
    values, names = [], []
    for column in CONTINUOUS + BINARY:
        values.append(frame[column].fillna(prep['impute'][column]).to_numpy(float))
        names.append(column)
    for column in NOMINAL:
        raw = frame[column].fillna(prep['impute'][column]).to_numpy(float)
        for level in prep['categories'][column]:
            values.append((raw == level).astype(float))
            names.append(f'{column}_level_{int(level)}')
    return np.column_stack(values), names

def score(y, p):
    result = metrics(y, p)
    order = np.argsort(-p, kind='stable')
    labels, probabilities = y[order], p[order]
    ends = np.r_[np.flatnonzero(probabilities[:-1] != probabilities[1:]), len(y) - 1]
    tp = np.cumsum(labels)[ends]
    recall = tp / y.sum()
    precision = tp / (ends + 1)
    result['average_precision'] = float(np.sum(np.diff(np.r_[0., recall]) * precision))
    result['predicted_positive_n'] = int(np.sum(p >= .5))
    return result

def fit_logistic(X, y, ridge=1.):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.
    z = (X - mu) / sd
    def objective(w):
        logits = w[0] + z @ w[1:]
        residual = expit(logits) - y
        value = np.logaddexp(0., logits).sum() - y @ logits + .5 * ridge * (w[1:] @ w[1:])
        return value, np.r_[residual.sum(), z.T @ residual + ridge * w[1:]]
    fitted = minimize(objective, np.zeros(X.shape[1] + 1), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    return fitted.x, mu, sd

def lr_predict(model, X):
    w, mu, sd = model
    return expit(w[0] + ((X - mu) / sd) @ w[1:])

def plan_for(X, bounds, names, soft):
    left, matched = engine.match_rules(X, bounds)
    right = left.copy()
    fraction = np.zeros_like(left, dtype=float)
    offset, pairs = 0, []
    for j, boundary in enumerate(bounds):
        count = len(boundary) - 1
        if soft and names[j] in CONTINUOUS:
            # Fixed endpoint extension applies only to continuous inputs.
            if count > 1:
                centers = (boundary[:-1] + boundary[1:]) / 2
                k = np.clip(np.searchsorted(centers, X[:, j], side='right') - 1, 0, count - 2)
                fraction[:, j] = np.clip((X[:, j] - centers[k]) / (centers[k + 1] - centers[k]), 0, 1)
                left[:, j], right[:, j] = offset + k, offset + k + 1
                pairs.extend((offset + i, offset + i + 1) for i in range(count - 1))
        offset += count
    if soft:
        matched = np.ones(len(X), dtype=bool)
        for j, boundary in enumerate(bounds):
            if names[j] not in CONTINUOUS:
                matched &= (X[:, j] >= boundary[0]) & (X[:, j] <= boundary[-1])
    return left, right, fraction, matched, np.asarray(pairs, dtype=int).reshape(-1, 2)

def predict_plan(model, plan):
    left, right, fraction, matched, _ = plan
    t = fraction[:, :, None]
    beta = (1 - t) * model.beta[left] + t * model.beta[right]
    r = (1 - fraction) * model.r[left] + fraction * model.r[right]
    w = (1 - fraction) * model.w[left] + fraction * model.w[right]
    p = engine.archived.er_rule_batch(r, w, beta)[:, 1]
    return np.where(matched & np.isfinite(p), p, 0.), matched

def rule_fit(X, y, names, seed, soft, strength, directory, max_intervals=20, custom_boundaries=None, sigma0=.5):
    initial = engine.build_ibrb(X, y, optimize=False, max_intervals=max_intervals, n_min=5,
                               custom_boundaries=custom_boundaries)
    L = initial.L
    plan = plan_for(X, initial.boundaries, names, soft)
    assert plan[3].all()
    def objective(z):
        model = engine.IBRBModel(initial.boundaries, z[:2*L].reshape(L, 2), z[2*L:3*L], z[3*L:])
        p, _ = predict_plan(model, plan)
        parameters = np.column_stack([model.beta, model.r, model.w])
        pairs = plan[4]
        penalty = float(np.mean((parameters[pairs[:, 0]] - parameters[pairs[:, 1]])**2)) if len(pairs) else 0.
        return float(np.mean((p - y)**2)) + strength * penalty
    z, info = engine.cmaes_min(initial.initial_parameters_, objective, L, seed, generations=200, sigma0=sigma0)
    model = engine.IBRBModel(initial.boundaries, z[:2*L].reshape(L, 2), z[2*L:3*L], z[3*L:])
    info['objective_name'] = 'training MSE + fixed lambda neighbor smoothness'
    info['lambda'] = strength
    info['soft_matching'] = soft
    info['max_intervals'] = max_intervals
    p, _ = predict_plan(model, plan)
    info['raw_training_mse'] = float(np.mean((p-y)**2))
    directory.mkdir(exist_ok=True)
    np.savez(directory / '模型.npz', beta=model.beta, r=model.r, w=model.w, initial_parameters=initial.initial_parameters_)
    write_json(directory / '边界.json', [b.tolist() for b in model.boundaries])
    write_json(directory / '训练信息.json', info)
    return model, info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shap-objective', choices=['quantile', 'binary'], default='quantile')
    args = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    out = HERE / f'NLST训练内可行性评估_{stamp}'
    out.mkdir()
    cohort = ROOT / '数据来源核查' / 'NLST患者级候选_20261005'
    membership = pd.read_csv(cohort / '候选划分成员表.csv')
    source = pd.read_csv(cohort / '患者级建模候选.csv')
    # Filter the source before feature transformation; fixed-test rows are never transformed or scored.
    ids = {role: membership.loc[membership[f'in_{role}'], 'pid'].tolist() for role in ['SHAP', 'IBRB', 'validation']}
    allowed = set(ids['SHAP']) | set(ids['validation'])
    source = source[source.pid.isin(allowed)].set_index('pid')
    frames = {role: source.loc[pid].copy() for role, pid in ids.items()}
    assert set(ids['IBRB']) <= set(ids['SHAP'])
    assert not set(ids['validation']) & set(ids['SHAP'])
    write_json(out / '运行前固定方案.json', {
        'source_directory': str(cohort), 'roles': {k: {'n': len(v), 'positive': int(v.label_365d.sum())} for k,v in frames.items()},
        'SHAP': {'objective': args.shap_objective, 'alpha_if_quantile': .5, 'rounds': 200,
                 'target_standardization': args.shap_objective == 'quantile', 'selection': 'top7 encoded features by mean absolute training SHAP'},
        'preprocessing': 'medians/modes and nominal one-hot vocabulary fit on SHAP1000 training only; shared across paired models',
        'rule_settings': {'max_intervals': 20, 'n_min': 5, 'CMA_generations': 200, 'sigma0': .5, 'seeds': SEEDS},
        'arms': ['hard_MSE', 'soft_lambda0', 'soft_lambda0.05'],
        'baselines': ['LR260_all', 'LR260_same7', 'LR1000_all', 'LR1000_same7', 'training_prevalence_constant'],
        'threshold': .5, 'validation_n': 300, 'fixed_test_evaluated': False,
        'selection': 'bounded feasibility comparison; no automatic manuscript adoption or new lambda grid',
        'interpretation': 'LR260 uses260 labels and shared unsupervised preprocessing; same7 also inherits SHAP1000 feature selection; LR1000 matches upstream label count',
    })
    prep = preprocess_fit(frames['SHAP'])
    write_json(out / '训练内预处理.json', prep)
    matrices = {}
    labels = {}
    for role, frame in frames.items():
        matrices[role], names = transform(frame, prep)
        labels[role] = frame.label_365d.to_numpy(int)
        frame[['label_365d']].reset_index().to_csv(out / f'{role}_成员.csv', index=False)
    X, y = matrices['SHAP'], labels['SHAP']
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.
    ym, ys = float(y.mean()), float(y.std())
    params = {'objective': args.shap_objective, 'seed': SEEDS[0], 'num_threads': 2,
              'verbosity': -1, 'deterministic': True, 'force_col_wise': True}
    target = y
    if args.shap_objective == 'quantile':
        params['alpha'] = .5
        target = (y-ym)/ys
    base = lgb.train(params, lgb.Dataset((X-mu)/sd, label=target, feature_name=names), num_boost_round=200)
    (out / f'SHAP模型_{args.shap_objective}.txt').write_text(base.model_to_string(), encoding='utf-8')
    contributions = base.predict((X-mu)/sd, pred_contrib=True)
    prediction = base.predict((X-mu)/sd, raw_score=True)
    np.testing.assert_allclose(contributions.sum(1), prediction, atol=1e-9, rtol=1e-9)
    importance = np.abs(contributions[:, :-1]).mean(0)
    eligible_features = np.flatnonzero(matrices['IBRB'].std(0) > 0)
    selected = eligible_features[np.argsort(-importance[eligible_features], kind='stable')[:7]]
    selected_names = [names[i] for i in selected]
    pd.DataFrame({'encoded_feature': names, 'mean_absolute_SHAP': importance,
                  'variable_in_IBRB260': matrices['IBRB'].std(0) > 0,
                  'selected': [i in selected for i in range(len(names))]}).sort_values('mean_absolute_SHAP', ascending=False).to_csv(out/'SHAP排序.csv', index=False)
    write_json(out / 'SHAP诊断.json', {'encoded_feature_count': len(names), 'selected': selected_names,
        'nonzero_importance_features': int(np.sum(importance > 1e-12)), 'SHAP_importance_sum': float(importance.sum()),
        'training_prediction_std': float(prediction.std()), 'tree_count': base.num_trees(),
        'target_mean': ym, 'target_sd': ys, 'objective': args.shap_objective,
        'SHAP_output_unit': 'raw log odds' if args.shap_objective == 'binary' else 'standardized quantile output',
        'uninformative_ranking': bool(np.max(importance) <= 1e-12)})
    np.savez(out/'SHAP标准化.npz', mean=mu, sd=sd, target_mean=ym, target_sd=ys, selected=selected)
    print('SHAP selected '+str(selected_names)+' sum='+str(importance.sum()), flush=True)
    records = []
    def evaluate(name, seed, train_role, train_p, validation_p, rules=None, unmatched=0):
        for role, p in [(train_role, train_p), ('validation', validation_p)]:
            path = out / f'{name}_seed{seed}_{role}_预测.csv'
            pd.DataFrame({'pid': ids[role], 'y_true': labels[role], 'score': p}).to_csv(path, index=False)
            records.append({'model': name, 'seed': seed, 'subset': role, 'rules': rules,
                            'unmatched_n': unmatched if role == 'validation' else 0, 'prediction_file': str(path),
                            **score(labels[role], p)})
    for train_role, n in [('IBRB', 260), ('SHAP', 1000)]:
        for suffix, columns in [('all', np.arange(len(names))), ('same7', selected)]:
            model = fit_logistic(matrices[train_role][:, columns], labels[train_role])
            name = f'LR{n}_{suffix}'
            np.savez(out/f'{name}.npz', weights=model[0], mean=model[1], sd=model[2], columns=columns)
            evaluate(name, 0, train_role, lr_predict(model, matrices[train_role][:, columns]), lr_predict(model, matrices['validation'][:, columns]))
    prevalence = float(labels['IBRB'].mean())
    evaluate('training_prevalence_constant', 0, 'IBRB', np.full(260, prevalence), np.full(300, prevalence))
    for name, soft, strength in [('hard_MSE', False, 0.), ('soft_lambda0', True, 0.), ('soft_lambda0.05', True, .05)]:
        for seed in SEEDS:
            start = time.monotonic()
            model, info = rule_fit(matrices['IBRB'][:, selected], labels['IBRB'], selected_names, seed, soft, strength, out/f'{name}_seed{seed}')
            training_p, _ = predict_plan(model, plan_for(matrices['IBRB'][:, selected], model.boundaries, selected_names, soft))
            validation_p, matched = predict_plan(model, plan_for(matrices['validation'][:, selected], model.boundaries, selected_names, soft))
            if not soft:
                reference_p, reference_matched = engine.predict_proba(model, matrices['validation'][:, selected])
                np.testing.assert_allclose(reference_p, validation_p, atol=1e-12)
                np.testing.assert_array_equal(reference_matched, matched)
            evaluate(name, seed, 'IBRB', training_p, validation_p, model.L, int((~matched).sum()))
            print(f'READY {name} seed={seed} L={model.L} elapsed={time.monotonic()-start:.1f}s validation={score(labels["validation"], validation_p)}', flush=True)
            pd.DataFrame(records).to_csv(out/'全部训练内指标.csv', index=False)
    result = pd.DataFrame(records)
    result.to_csv(out/'全部训练内指标.csv', index=False)
    result[result.subset.eq('validation')].groupby('model')[['accuracy','mse','auc','average_precision','recall','specificity','f1','unmatched_n']].agg(['mean','std']).to_csv(out/'开发验证均值与标准差.csv')
    write_json(out/'完成记录.json', {'completed': True, 'IBRB_fits': 9, 'LR_fits': 4, 'fixed_test_evaluated': False,
                                    'metric_rows': len(result), 'selected_features': selected_names, 'SHAP_objective': args.shap_objective})
    (HERE/'最新NLST训练内可行性目录.txt').write_text(str(out), encoding='utf-8')
    print('COMPLETE '+str(out), flush=True)

if __name__ == '__main__':
    main()
