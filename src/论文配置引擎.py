# -*- coding: utf-8 -*-
"""IBRB reproduction of manuscript Algorithm 1 and P213--216.

The archival engine remains unchanged. ER and interval matching are reused;
partitioning, initialization and CMA generation accounting are explicit here.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ibrb_engine as archived

IBRBModel = archived.IBRBModel
predict_proba = archived.predict_proba
match_rules = archived.match_rules


def supervised_boundaries(values, labels, max_intervals=20, n_min=5):
    # paper mode enumerates all midpoints and stops at nonpositive gain.
    padded = archived.info_gain_split(values, labels, max_intervals,
                                     mode="paper", n_min=n_min)
    return np.unique(padded)


def project_parameters(candidate, L, lower=1e-4, upper=1.0):
    """Least-squares projection onto the bounded binary belief simplex."""
    x = np.asarray(candidate, dtype=float).copy()
    was_vector = x.ndim == 1
    if was_vector:
        x = x[:, None]
    beta = x[:2 * L].reshape(L, 2, -1)
    first = np.clip((1.0 + beta[:, 0] - beta[:, 1]) / 2.0,
                    max(lower, 1.0 - upper), min(upper, 1.0 - lower))
    beta[:, 0] = first
    beta[:, 1] = 1.0 - first
    x[2 * L:] = np.clip(x[2 * L:], lower, upper)
    return x[:, 0] if was_vector else x


def cmaes_min(x0, fitness, L, seed, generations=200, sigma0=.5,
              lower=1e-4, upper=1.0, population_multiplier=1,
              initial_std=None, projector=None):
    """Positive-weight CMA-ES with projected candidates and train-only selection.

    Default population and covariance equations follow the archived algorithm.
    An explicit multiplier supports bounded optimizer-budget comparisons.
    Optional coordinate standard deviations set the initial diagonal covariance;
    default unit scales retain the original trajectory.
    Evaluation counts increment by population size; generations have their own
    loop counter. This fixes the mixed counter units in archival cmaes.m.
    """
    # A supplied projector supports equivalent effective coordinates. The
    # default retains the archived full-coordinate projection exactly.
    project = projector if projector is not None else lambda z: project_parameters(z, L, lower, upper)
    rng = np.random.default_rng(seed)
    mean = project(x0)
    n = len(mean)
    population = int((10 + int(3 * np.log(n))) * population_multiplier)
    mu = population // 2
    weights = np.log(mu + .5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) /
              ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    sigma = float(sigma0)
    pc, ps = np.zeros(n), np.zeros(n)
    B, D = np.eye(n), np.ones(n)
    C, inverse_sqrt = np.eye(n), np.eye(n)
    if initial_std is not None:
        initial_std = np.asarray(initial_std, dtype=float)
        if initial_std.shape != (n,) or not np.all(initial_std > 0):
            raise ValueError("initial_std must contain one positive scale per parameter")
        D = initial_std.copy()
        C, inverse_sqrt = np.diag(D**2), np.diag(1.0/D)
    chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    evaluations, last_eigen = 0, 0
    best_x, best_f = mean.copy(), float(fitness(mean))
    initial_f = best_f
    trace = []
    for generation in range(1, generations + 1):
        proposals = mean[:, None] + sigma * (
            B @ (D[:, None] * rng.standard_normal((n, population))))
        candidates = project(proposals)
        values = np.array([fitness(candidates[:, k])
                           for k in range(population)])
        evaluations += population
        order = np.argsort(values, kind="stable")
        if values[order[0]] < best_f:
            best_f = float(values[order[0]])
            best_x = candidates[:, order[0]].copy()
        old = mean
        mean = candidates[:, order[:mu]] @ weights
        displacement = (mean - old) / sigma
        ps = ((1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) *
              (inverse_sqrt @ displacement))
        hsig = ((ps @ ps) /
                (1 - (1 - cs) ** (2 * evaluations / population)) / n
                < 2 + 4 / (n + 1))
        pc = ((1 - cc) * pc + float(hsig) *
              np.sqrt(cc * (2 - cc) * mueff) * displacement)
        steps = (candidates[:, order[:mu]] - old[:, None]) / sigma
        C = ((1 - c1 - cmu) * C + c1 * (
            np.outer(pc, pc) + (1 - float(hsig)) * cc * (2 - cc) * C)
            + cmu * (steps * weights) @ steps.T)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi - 1))
        if evaluations - last_eigen > population / (c1 + cmu) / n / 10:
            last_eigen = evaluations
            C = np.triu(C) + np.triu(C, 1).T
            eigenvalues, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigenvalues, 1e-20))
            inverse_sqrt = (B * (1 / D)) @ B.T
        trace.append({"generation": generation,
                      "candidate_evaluations": evaluations,
                      "best_training_mse": best_f, "sigma": float(sigma)})
    return best_x, {
        "training_mse": best_f, "initial_training_mse": initial_f,
        "generations": generations, "population": population,
        "candidate_evaluations": evaluations,
        "initial_evaluations": 1, "dimension": n,
        "sigma0": float(sigma0), "population_multiplier": population_multiplier,
        **({"initial_coordinate_std": initial_std.tolist()}
           if initial_std is not None else {}),
        "retained_initial_model": bool(best_f == initial_f), "trace": trace}


def build_ibrb(X, y, seed=None, method="supervised", optimize=True,
               max_intervals=20, n_min=5, generations=200,
               custom_boundaries=None):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    if custom_boundaries is not None:
        bounds = [np.asarray(b, dtype=float) for b in custom_boundaries]
    elif method == "supervised":
        bounds = [supervised_boundaries(X[:, m], y, max_intervals, n_min)
                  for m in range(X.shape[1])]
    elif method == "equal_width":
        bounds = [archived.equal_width_split(X[:, m], max_intervals)
                  for m in range(X.shape[1])]
    else:
        raise ValueError(method)
    L = sum(len(b) - 1 for b in bounds)
    beta = archived.laplace_init(X, y, bounds)
    x0 = np.r_[beta.ravel(), np.full(L, .9), np.full(L, 1.0 / L)]
    fitness = archived.make_fitness(X, y, bounds, L, 2)
    if optimize:
        chosen, info = cmaes_min(x0, fitness, L, seed, generations)
    else:
        chosen = x0
        f = float(fitness(chosen))
        info = {"training_mse": f, "initial_training_mse": f,
                "generations": 0, "population": 0,
                "candidate_evaluations": 0, "initial_evaluations": 1,
                "dimension": 4 * L, "retained_initial_model": True,
                "trace": []}
    model = IBRBModel(bounds, chosen[:2 * L].reshape(L, 2),
                      chosen[2 * L:3 * L], chosen[3 * L:])
    model.train_mse_ = info["training_mse"]
    model.fit_info_ = info
    model.initial_parameters_ = x0
    return model
