from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import 论文配置引擎 as engine
run = SimpleNamespace(engine=engine)
CONTINUOUS = ['age','baseline_nodule_count','index_long_diameter_mm','index_perpendicular_diameter_mm']
def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


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
