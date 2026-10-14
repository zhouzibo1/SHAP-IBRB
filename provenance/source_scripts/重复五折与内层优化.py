# -*- coding: utf-8 -*-
"""Prespecified repeated CV; conditional bounded tuning evaluated in outer folds."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / '依赖'))
import json
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import t
import 论文配置引擎 as eng
from 最小核验实验 import metrics, write_json
from 三组最小消融 import Tee
from 三次分层划分验证 import standardize_fit
from 训练内验证改进 import split_train
from 已保存预测统计补充 import holm

OUTER_SEEDS = [20262041, 20262042]
GRID = [(k, s) for k in [5, 10, 20] for s in [.15, .5]]
BASELINES = ['B_no_IG', 'D_matched_equal', 'LR_1000', 'LGBM_1000']


def fold_ids(y, n_folds, seed):
    rng = np.random.default_rng(seed)
    ids = np.empty(len(y), dtype=int)
    # Offset the larger class so total fold sizes stay balanced.
    offset = 0
    for c in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == c))
        ids[indices] = (np.arange(len(indices)) + offset) % n_folds
        offset = (offset + len(indices)) % n_folds
    return ids


def subset(rows, y, n, seed):
    _, local = split_train(y[rows], n, seed)
    return rows[local]


def shap_fit(X, y, seed, directory):
    directory.mkdir(parents=True, exist_ok=True)
    mu, sd, z = standardize_fit(X)
    ym, ys = float(y.mean()), float(y.std())
    base = lgb.train({'objective': 'quantile', 'alpha': .5, 'seed': seed,
                     'num_threads': 2, 'verbosity': -1, 'deterministic': True,
                     'force_col_wise': True}, lgb.Dataset(z, label=(y-ym)/ys),
                     num_boost_round=200)
    values = base.predict(z, pred_contrib=True)
    np.testing.assert_allclose(values.sum(1), base.predict(z), atol=1e-10, rtol=1e-10)
    importance = np.abs(values[:, :-1]).mean(0)
    columns = np.sort(np.argsort(-importance, kind='stable')[:7])
    (directory/'SHAP模型.txt').write_text(base.model_to_string(), encoding='utf-8')
    np.savez(directory/'标准化.npz', mean=mu, sd=sd, target_mean=ym, target_sd=ys)
    write_json(directory/'排序.json', {'importance': importance.tolist(), 'columns': columns.tolist()})
    return columns


def rule_fit(X, y, seed, k=20, sigma=.5, method='supervised', bounds=None):
    start = eng.build_ibrb(X, y, method=method, max_intervals=k,
                           optimize=False, custom_boundaries=bounds)
    L = start.L
    fitness = eng.archived.make_fitness(X, y, start.boundaries, L, 2)
    x, info = eng.cmaes_min(start.initial_parameters_, fitness, L, seed,
                            generations=200, sigma0=sigma)
    model = eng.IBRBModel(start.boundaries, x[:2*L].reshape(L, 2), x[2*L:3*L], x[3*L:])
    model.fit_info_ = info
    model.initial_parameters_ = start.initial_parameters_
    np.testing.assert_allclose(model.beta.sum(1), 1, atol=1e-13)
    return model


def save_rule(directory, model, columns, settings):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory/'模型.npz', beta=model.beta, r=model.r, w=model.w,
             initial_parameters=model.initial_parameters_)
    write_json(directory/'边界.json', [b.tolist() for b in model.boundaries])
    write_json(directory/'配置.json', {'columns': columns.tolist(), **settings,
                                    'fit_info': model.fit_info_})


def lr_fit(X, y, penalty):
    mu, sd, z = standardize_fit(X)
    def loss(w):
        logits = w[0] + z@w[1:]
        value = np.logaddexp(0, logits).sum()-y@logits + .5*penalty*(w[1:]@w[1:])
        residual = expit(logits)-y
        return value, np.r_[residual.sum(), z.T@residual+penalty*w[1:]]
    result = minimize(loss, np.zeros(z.shape[1]+1), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    assert result.success, result.message
    return result.x, mu, sd


def lr_predict(model, X):
    w, mu, sd = model
    return expit(w[0]+((X-mu)/sd)@w[1:])


def gb_fit(X, y, seed, leaves, reg):
    mu, sd, z = standardize_fit(X)
    base = lgb.train({'objective': 'binary', 'learning_rate': .05, 'max_depth': 6,
                     'num_leaves': leaves, 'min_data_in_leaf': 20, 'lambda_l2': reg,
                     'seed': seed, 'num_threads': 2, 'verbosity': -1,
                     'deterministic': True, 'force_col_wise': True},
                     lgb.Dataset(z, label=y), num_boost_round=200)
    return base, mu, sd


def gb_predict(model, X):
    base, mu, sd = model
    return base.predict((X-mu)/sd)


def tuned_baselines(X, y, inner, seed, directory):
    lr_grid, gb_grid = [.1, 1., 10.], [(15, 1.), (31, 1.), (15, 5.)]
    rows = []
    for f in range(3):
        tr, va = np.flatnonzero(inner != f), np.flatnonzero(inner == f)
        for penalty in lr_grid:
            p = lr_predict(lr_fit(X[tr], y[tr], penalty), X[va])
            rows.append({'model': 'LR', 'candidate': str(penalty), 'fold': f+1,
                         'mse': float(np.mean((p-y[va])**2))})
        for leaves, reg in gb_grid:
            p = gb_predict(gb_fit(X[tr], y[tr], seed+f, leaves, reg), X[va])
            rows.append({'model': 'LGBM', 'candidate': f'{leaves},{reg}', 'fold': f+1,
                         'mse': float(np.mean((p-y[va])**2))})
    frame = pd.DataFrame(rows)
    frame.to_csv(directory/'基线内层搜索.csv', index=False)
    means = frame.groupby(['model', 'candidate']).mse.mean()
    lr_best = min(lr_grid, key=lambda c: means.loc['LR', str(c)])
    gb_best = min(gb_grid, key=lambda c: means.loc['LGBM', f'{c[0]},{c[1]}'])
    lr = lr_fit(X, y, lr_best)
    gb = gb_fit(X, y, seed, *gb_best)
    np.savez(directory/'逻辑回归.npz', weights=lr[0], mean=lr[1], sd=lr[2])
    (directory/'二分类LightGBM.txt').write_text(gb[0].model_to_string(), encoding='utf-8')
    np.savez(directory/'二分类LightGBM标准化.npz', mean=gb[1], sd=gb[2])
    write_json(directory/'基线选择.json', {'LR_penalty': lr_best, 'LGBM_leaves_L2': gb_best})
    return lr, gb


def comparisons(frame, focal, controls):
    result = []
    for other in controls:
        a = frame[frame.variant == focal].sort_values(['repeat', 'fold'])
        b = frame[frame.variant == other].sort_values(['repeat', 'fold'])
        assert np.array_equal(a[['repeat', 'fold']].to_numpy(), b[['repeat', 'fold']].to_numpy())
        for metric in ['accuracy', 'mse', 'auc']:
            delta = a[metric].to_numpy()-b[metric].to_numpy()
            mean, variance, n = float(delta.mean()), float(delta.var(ddof=1)), len(delta)
            row = {'comparison': f'{focal} minus {other}', 'metric': metric,
                   'difference': mean, 'fold_difference_sd': variance**.5, 'df': n-1,
                   'independent_source_records': 5000, 'paired_outer_evaluations': n}
            for key, ratio in [('actual_label_budget', 1.), ('outer_pool_sensitivity', .25)]:
                se = np.sqrt((1/n+ratio)*variance)
                stat = mean/se if se else (0. if mean == 0 else np.copysign(np.inf, mean))
                p = float(2*t.sf(abs(stat), n-1))
                width = float(t.ppf(.975, n-1)*se)
                row.update({f'{key}_p': p, f'{key}_se': float(se),
                            f'{key}_ci_low': mean-width, f'{key}_ci_high': mean+width})
            result.append(row)
    result = pd.DataFrame(result)
    for key in ['actual_label_budget', 'outer_pool_sensitivity']:
        result[f'{key}_holm'] = holm(result[f'{key}_p'])
    return result


def optimize_inner(X, y, rows, seed, directory):
    inner = fold_ids(y[rows], 3, seed+200)
    scores = []
    for f in range(3):
        tr, va = rows[inner != f], rows[inner == f]
        rule = subset(tr, y, 260, seed+300+f)
        d = directory/f'inner{f+1}'
        columns = shap_fit(X[tr], y[tr], seed+f, d)
        pd.DataFrame({'source_row_1based': np.r_[tr, va]+1,
                      'subset': ['SHAP_train']*len(tr)+['validation']*len(va)}).to_csv(d/'成员.csv', index=False)
        pd.DataFrame({'source_row_1based': rule+1}).to_csv(d/'规则训练成员.csv', index=False)
        assert not np.intersect1d(tr, va).size and np.isin(rule, tr).all()
        for k, sigma in GRID:
            start = time.perf_counter()
            model = rule_fit(X[rule][:, columns], y[rule], seed+500+f, k, sigma)
            cd = d/f'K{k}_sigma{sigma}'
            save_rule(cd, model, columns, {'max_intervals': k, 'sigma0': sigma})
            p, matched = eng.predict_proba(model, X[va][:, columns])
            pd.DataFrame({'source_row_1based': va+1, 'y_true': y[va], 'raw_score': p,
                          'matched': matched}).to_csv(cd/'内层预测.csv', index=False)
            row = {'inner_fold': f+1, 'K': k, 'sigma0': sigma, 'mse': float(np.mean((p-y[va])**2)),
                   'rules': model.L, 'elapsed_seconds': time.perf_counter()-start}
            scores.append(row)
            pd.DataFrame(scores).to_csv(directory/'规则内层搜索.csv', index=False)
            print(f'INNER {directory.name} fold={f+1} K={k} sigma={sigma} MSE={row["mse"]:.5f}', flush=True)
    frame = pd.DataFrame(scores)
    mean = frame.groupby(['K', 'sigma0']).mse.mean()
    # Stable grid order breaks exact ties without observing outer predictions.
    selected = min(GRID, key=lambda c: mean.loc[c])
    write_json(directory/'规则内层选择.json', {'K': selected[0], 'sigma0': selected[1],
                                            'inner_mean_mse': float(mean.loc[selected])})
    return selected


def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    out = HERE/f'重复五折与内层优化_{stamp}'
    out.mkdir()
    (HERE/'最新交叉验证目录.txt').write_text(str(out), encoding='utf-8')
    sys.stdout = Tee(sys.stdout, out/'运行日志.txt')
    (out/'运行前固定方案.md').write_text((HERE/'重复五折与内层优化方案.md').read_text(encoding='utf-8'), encoding='utf-8')
    data = pd.read_csv(ROOT/'Lung Cancer Dataset.csv')
    X, y = data.iloc[:, :-1].to_numpy(float), data.iloc[:, -1].to_numpy(int)
    assert len(y) == 5000
    splits, member_rows = [], []
    for repeat, outer_seed in enumerate(OUTER_SEEDS, 1):
        ids = fold_ids(y, 5, outer_seed)
        for f in range(5):
            tr, te = np.flatnonzero(ids != f), np.flatnonzero(ids == f)
            seed = outer_seed+1000+f
            shap = subset(tr, y, 1000, seed)
            rule = subset(shap, y, 260, seed+100)
            inner = fold_ids(y[shap], 3, seed+200)
            assert len(te) == 1000 and not np.intersect1d(shap, te).size and np.isin(rule, shap).all()
            d = out/f'repeat{repeat}_fold{f+1}'
            d.mkdir()
            pd.DataFrame({'source_row_1based': shap+1, 'inner_fold': inner+1}).to_csv(d/'基线内层成员.csv', index=False)
            for role, rows in [('outer_train', tr), ('test', te), ('SHAP_train', shap), ('rule_train', rule)]:
                member_rows.extend({'repeat': repeat, 'fold': f+1, 'subset': role,
                                    'source_row_1based': int(r+1)} for r in rows)
            splits.append((repeat, f+1, seed, shap, rule, te, inner, d))
    pd.DataFrame(member_rows).to_csv(out/'预先固定成员.csv', index=False)
    write_json(out/'预先固定配置.json', {'outer_seeds': OUTER_SEEDS, 'candidate_grid': GRID,
              'outer_folds': 10, 'outer_test_each': 1000, 'rule_training_each': 260,
              'SHAP_training_each': 1000, 'conditional_optimization_trigger': 'A-B MSE corrected Holm >= .05',
              'maximum_optimized_inner_rule_fits': 180, 'baseline_inner_folds': 3,
              'scope': 'exploratory repeated CV on previously investigated benchmark'})
    print('OUTPUT '+str(out), flush=True)
    models = []
    for repeat, fold, seed, shap, rule, te, inner, d in splits:
        print(f'FIT original repeat={repeat} fold={fold}', flush=True)
        columns = shap_fit(X[shap], y[shap], seed, d/'SHAP')
        bounds = [eng.supervised_boundaries(X[rule, c], y[rule]) for c in columns]
        matched = [eng.archived.equal_width_split(X[rule, c], len(b)-1) for c, b in zip(columns, bounds)]
        for group, method, custom in [('A_full', 'supervised', bounds),
                                      ('B_no_IG', 'equal_width', None),
                                      ('D_matched_equal', 'equal_width', matched)]:
            model = rule_fit(X[rule][:, columns], y[rule], seed, method=method, bounds=custom)
            save_rule(d/group, model, columns, {'training_n': 260, 'label_access_n': 1000,
                                             'max_intervals': 20, 'sigma0': .5, 'seed': seed})
            models.append((repeat, fold, group, model, columns, te, d, 'rule'))
        lr, gb = tuned_baselines(X[shap], y[shap], inner, seed, d)
        models.extend([(repeat, fold, 'LR_1000', lr, None, te, d, 'lr'),
                       (repeat, fold, 'LGBM_1000', gb, None, te, d, 'gb')])
    write_json(out/'原配置评价前锁定.json', {'saved_outer_models': len(models), 'all_original_fits_finished': True})
    results, checks = [], []
    def evaluate(entries):
        for repeat, fold, group, model, columns, te, d, kind in entries:
            if kind == 'rule':
                p, matched = eng.predict_proba(model, X[te][:, columns])
                saved = np.load(d/group/'模型.npz')
                bounds = [np.array(b) for b in json.loads((d/group/'边界.json').read_text(encoding='utf-8'))]
                loaded = eng.IBRBModel(bounds, saved['beta'], saved['r'], saved['w'])
                q, _ = eng.predict_proba(loaded, X[te][:, columns])
                np.testing.assert_array_equal(p, q)
            elif kind == 'lr':
                p, matched = lr_predict(model, X[te]), np.ones(len(te), bool)
                saved = np.load(d/'逻辑回归.npz')
                np.testing.assert_array_equal(p, lr_predict((saved['weights'], saved['mean'], saved['sd']), X[te]))
            else:
                p, matched = gb_predict(model, X[te]), np.ones(len(te), bool)
                saved = np.load(d/'二分类LightGBM标准化.npz')
                loaded = lgb.Booster(model_str=(d/'二分类LightGBM.txt').read_text(encoding='utf-8'))
                np.testing.assert_array_equal(p, gb_predict((loaded, saved['mean'], saved['sd']), X[te]))
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            (d/group).mkdir(exist_ok=True)
            pd.DataFrame({'source_row_1based': te+1, 'y_true': y[te], 'raw_score': p,
                          'prediction': (p >= .5).astype(int), 'matched': matched}).to_csv(d/group/'测试预测.csv', index=False)
            row = {'repeat': repeat, 'fold': fold, 'variant': group, 'matched_test_n': int(matched.sum()),
                   'rules': model.L if kind == 'rule' else None, **metrics(y[te], p)}
            results.append(row)
            checks.append({'repeat': repeat, 'fold': fold, 'variant': group, 'reload_identical': True})
    evaluate(models)
    frame = pd.DataFrame(results)
    frame.to_csv(out/'原配置全部指标.csv', index=False)
    original = comparisons(frame, 'A_full', BASELINES)
    original.to_csv(out/'原配置校正比较.csv', index=False)
    trigger = original[(original.comparison == 'A_full minus B_no_IG') & (original.metric == 'mse')].iloc[0]
    need_opt = not (trigger.difference < 0 and trigger.actual_label_budget_holm < .05)
    write_json(out/'优化分支判定.json', {'primary_difference': float(trigger.difference),
               'primary_adjusted_p': float(trigger.actual_label_budget_holm), 'run_bounded_optimization': bool(need_opt)})
    print(f'ORIGINAL MSE A-B delta={trigger.difference:.6f} Holm={trigger.actual_label_budget_holm:.6f} optimize={need_opt}', flush=True)
    if need_opt:
        optimized = []
        for repeat, fold, seed, shap, rule, te, inner, d in splits:
            print(f'FIT optimized repeat={repeat} fold={fold}', flush=True)
            selection_dir = d/'优化内层'
            selection_dir.mkdir()
            k, sigma = optimize_inner(X, y, shap, seed, selection_dir)
            columns = np.array(json.loads((d/'SHAP'/'排序.json').read_text(encoding='utf-8'))['columns'])
            model = rule_fit(X[rule][:, columns], y[rule], seed, k=k, sigma=sigma)
            save_rule(d/'F_inner_tuned', model, columns, {'K': k, 'sigma0': sigma,
                       'training_n': 260, 'label_access_n': 1000, 'selection': '3-fold inner MSE'})
            optimized.append((repeat, fold, 'F_inner_tuned', model, columns, te, d, 'rule'))
        write_json(out/'优化变体评价前锁定.json', {'saved_optimized_outer_models': 10,
                   'all_optimized_fits_finished_before_optimized_outer_predictions': True})
        evaluate(optimized)
        frame = pd.DataFrame(results)
        improvement = comparisons(frame, 'F_inner_tuned', ['A_full']+BASELINES)
        improvement.to_csv(out/'优化变体校正比较.csv', index=False)
        combined = pd.concat([original, improvement], ignore_index=True)
        for key in ['actual_label_budget', 'outer_pool_sensitivity']:
            combined[f'{key}_joint_holm'] = holm(combined[f'{key}_p'])
        combined.to_csv(out/'全部比较联合校正.csv', index=False)
    frame.to_csv(out/'全部逐折指标.csv', index=False)
    summary = frame.groupby('variant')[['accuracy', 'mse', 'auc', 'recall', 'f1']].agg(['mean', 'std'])
    summary.to_csv(out/'均值与折间标准差.csv')
    for repeat in [1, 2]:
        members = pd.read_csv(out/'预先固定成员.csv')
        tested = members[(members.repeat == repeat) & (members.subset == 'test')].source_row_1based
        assert len(tested) == 5000 and tested.nunique() == 5000
    write_json(out/'完成核验.json', {'all_checks_passed': True, 'all_saved_models_reloaded': checks,
              'source_records': 5000, 'repeats': 2, 'test_predictions_per_model': 10000,
              'unique_test_records_per_model': 5000, 'optimization_executed': bool(need_opt),
              'new_rule_fits': 30+(190 if need_opt else 0), 'original_manuscript_modified': False})
    print(summary.to_string(), flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
