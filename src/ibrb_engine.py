# -*- coding: utf-8 -*-
"""
IBRB 引擎 Python 移植版（忠实对应 肺癌修改/*.m）
================================================================
对应关系：
  info_gain_split      <- fun_interval.m / funtest_interval.m 内嵌函数
  er_rule_batch        <- ER_Rule.m（向量化）
  cmaes_min            <- cmaes.m（Hansen purecmaes + 论文的等式约束投影算子）
  build_ibrb/predict   <- interval_range_brb.m 主流程

与 MATLAB 版唯一的算法差异（结果不变）：
  区间划分从适应度函数内部提到优化循环外——划分只依赖训练集，
  优化期间训练集固定，因此划分结果不变，速度提升约 lambda 倍。

mode 说明：
  "faithful"：完全复刻 MATLAB 版（候选点截取前 50 个、无 IG>0 停止条件、
              边界填充至 21 个参考点）
  "paper"   ：按论文 Algorithm 1 实现（增加 IG<=0 停止、K_max/n_min 可配）
"""

import numpy as np


# ---------------- 信息熵与区间划分 ----------------
def calc_entropy(labels):
    labels = np.asarray(labels)
    if len(labels) < 2:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / len(labels)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def info_gain_split(attr_values, labels, target_K, mode="faithful", n_min=5):
    """信息增益递归划分。返回 target_K+1 个参考点（升序）。"""
    attr_values = np.asarray(attr_values, dtype=float)
    labels = np.asarray(labels, dtype=float)
    valid = ~np.isnan(attr_values)
    va, vl = attr_values[valid], labels[valid]
    if len(va) < target_K + 1:
        return np.linspace(va.min(), va.max(), target_K + 1)

    current = [float(va.min()), float(va.max())]
    while len(current) - 1 < target_K:
        max_ig = -np.inf
        best_split = None
        best_idx = None
        for idx in range(len(current) - 1):
            left, right = current[idx], current[idx + 1]
            mask = (va >= left) & (va <= right)
            ia, il = va[mask], vl[mask]
            if len(il) < n_min:          # MATLAB 版硬编码 5
                continue
            sa = np.unique(ia)
            if len(sa) < 2:
                continue
            cand = (sa[:-1] + sa[1:]) / 2.0
            if mode == "faithful":
                cand = cand[:50]          # MATLAB: candidate_points(1:min(50,...))
            pe = calc_entropy(il)
            for sp in cand:
                lm = ia <= sp
                nl, nr = lm.sum(), len(ia) - lm.sum()
                if nl == 0 or nr == 0:
                    continue
                ce = (nl / len(ia)) * calc_entropy(il[lm]) + \
                     (nr / len(ia)) * calc_entropy(il[~lm])
                ig = pe - ce
                if ig > max_ig:
                    max_ig, best_split, best_idx = ig, sp, idx
        if best_split is None:
            break
        if mode == "paper" and max_ig <= 0:   # 论文 Algorithm 1 停止条件(3)
            break
        current = current[:best_idx + 1] + [best_split] + current[best_idx + 1:]

    fq = np.unique(current)
    if len(fq) < target_K + 1:                # 用最大值填充（MATLAB 兼容行为）
        fq = np.concatenate([fq, np.full(target_K + 1 - len(fq), fq[-1])])
    if len(fq) > target_K + 1:
        fq = fq[:target_K + 1]
    return np.sort(fq)


def equal_width_split(attr_values, target_K):
    """传统等距划分（对照组）。"""
    a = np.asarray(attr_values, dtype=float)
    a = a[~np.isnan(a)]
    return np.linspace(a.min(), a.max(), target_K + 1)


def mdlp_split(attr_values, labels, max_K=20):
    """MDLP 有监督离散化（Fayyad & Irani 1993，带 MDL 停止准则）。
    用于消融变体 E：验证本文划分的优势是否信息增益准则特有。
    实现：整体排序一次，候选点信息增益用累积和向量化计算。"""
    va = np.asarray(attr_values, dtype=float)
    vl = np.asarray(labels, dtype=float)
    ok = ~np.isnan(va)
    va, vl = va[ok], vl[ok]
    order = np.argsort(va, kind="mergesort")
    va, vl = va[order], vl[order].astype(int)

    def _ent(p):
        p = np.asarray(p, dtype=float)
        e = np.zeros_like(p)
        for q in (p, 1.0 - p):
            m = q > 0
            e[m] -= q[m] * np.log2(q[m])
        return e

    splits = []

    def rec(v, l):
        if len(splits) >= max_K - 1:
            return
        n = len(v)
        if n < 2 or l.min() == l.max():
            return
        cut = np.where(v[:-1] != v[1:])[0]       # 在 cut[i] 之后切开
        if len(cut) == 0:
            return
        cum = np.cumsum(l)
        tot = cum[-1]
        nl = cut + 1
        nr = n - nl
        pl = cum[cut] / nl
        pr = (tot - cum[cut]) / nr
        pe = float(_ent(np.array([tot / n]))[0])
        ig = pe - (nl / n) * _ent(pl) - (nr / n) * _ent(pr)
        bi = int(np.argmax(ig))
        best_ig = float(ig[bi])
        i = int(cut[bi])
        if best_ig <= 0:
            return
        # MDL 停止准则（Fayyad-Irani）
        l1, l2 = l[:i + 1], l[i + 1:]
        k = len(np.unique(l))
        k1 = len(np.unique(l1))
        k2 = len(np.unique(l2))
        e1 = float(_ent(np.array([l1.mean()]))[0])
        e2 = float(_ent(np.array([l2.mean()]))[0])
        delta = np.log(max(3 ** k - 2, 1)) - (k * pe - k1 * e1 - k2 * e2)
        if best_ig <= (np.log(max(n - 1, 1)) + delta) / n:
            return
        splits.append((v[i] + v[i + 1]) / 2.0)
        rec(v[:i + 1], l1)
        rec(v[i + 1:], l2)

    rec(va, vl)
    return np.array([va.min()] + sorted(splits) + [va.max()])


def laplace_init(X_train, y_train, boundaries, n_grades=2):
    """Laplace 平滑初始化规则置信度（对应论文 4.4 节所述初始化）。
    规则 k（属性 m 的第 j 个区间）的后件置信度：
        beta[k, c] = (count_c + 1) / (count_total + n_grades)
    其中 count_c 为训练集中该属性落入该区间的第 c 类样本数。"""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    L = sum(len(b) - 1 for b in boundaries)
    beta = np.zeros((L, n_grades))
    off = 0
    for m, b in enumerate(boundaries):
        v = X_train[:, m]
        for j in range(len(b) - 1):
            mask = (v >= b[j]) & (v <= b[j + 1])
            n = mask.sum()
            for c in range(n_grades):
                beta[off + j, c] = ((y_train[mask] == c).sum() + 1) / (n + n_grades)
        off += len(b) - 1
    return beta


# ---------------- ER 推理（向量化，对应 ER_Rule.m） ----------------
def er_rule_batch(r, w, p):
    """
    r: (S,7) 可靠度；w: (S,7) 权重；p: (S,7,N) 置信分布
    返回: (S,N) 融合后的置信度 P_fin
    """
    row_sum = p.sum(axis=2, keepdims=True)              # (S,7,1)
    base = (1 - r) + w * (1 - row_sum[..., 0])          # (S,7)
    sec2 = np.prod(base, axis=1)                        # (S,)
    sec3 = np.prod(1 - r, axis=1)                       # (S,)
    n_grade = p.shape[2]
    sec1 = np.stack([np.prod(base + w * p[:, :, i], axis=1)
                     for i in range(n_grade)], axis=1)  # (S,N)
    mu1 = sec1.sum(axis=1)
    mu2 = (n_grade - 1) * sec2
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = 1.0 / (mu1 - mu2)
        denom = 1 - mu * sec3
        p_fin = mu[:, None] * (sec1 - sec2[:, None]) / denom[:, None]
    return p_fin


# ---------------- 规则匹配 ----------------
def match_rules(X, boundaries):
    """
    对每个样本找每个属性第一个包含它的区间（与 MATLAB 嵌套循环 break 行为一致）。
    返回 rule_ids (S,M)、matched (S,)
    """
    S, M = X.shape
    rule_ids = np.zeros((S, M), dtype=int)
    matched = np.ones(S, dtype=bool)
    off = 0
    for m, b in enumerate(boundaries):
        v = X[:, m]
        idx = np.searchsorted(b, v, side="left") - 1
        idx = np.clip(idx, 0, len(b) - 2)
        ok = (v >= b[idx]) & (v <= b[idx + 1])
        rule_ids[:, m] = off + idx          # 累积偏移，兼容各特征区间数不同（消融 D/E）
        matched &= ok
        off += len(b) - 1
    return rule_ids, matched


# ---------------- 模型构建 / 预测 ----------------
class IBRBModel:
    def __init__(self, boundaries, beta, r, w, n_grades=2):
        self.boundaries = boundaries
        self.beta, self.r, self.w = beta, r, w
        self.N = n_grades
        self.L = beta.shape[0]


def predict_proba(model, X):
    """返回连续预测值（0~1，阳性等级的置信度）。未匹配样本按 MATLAB 行为置 0。"""
    X = np.asarray(X, dtype=float)
    S = X.shape[0]
    rule_ids, matched = match_rules(X, model.boundaries)
    p = model.beta[rule_ids]                  # (S,7,N)
    rr = model.r[rule_ids]                    # (S,7)
    ww = model.w[rule_ids]                    # (S,7)
    p_fin = er_rule_batch(rr, ww, p)          # (S,N)
    prob = p_fin[:, -1]                       # 效用 0/1 → 取阳性等级置信度
    prob = np.where(np.isfinite(prob), prob, 0.0)
    prob = np.where(matched, prob, 0.0)       # MATLAB: 未匹配样本预测值保持 0
    return prob, matched


# ---------------- 适应度函数 ----------------
def make_fitness(X_train, y_train, boundaries, L, N):
    """训练集 MSE 适应度。划分在外层完成，此处只做匹配+推理。"""
    rule_ids, matched = match_rules(X_train, boundaries)
    y = np.asarray(y_train, dtype=float)

    def fitness(x):
        beta = x[:L * N].reshape(L, N)
        r = x[L * N:L * N + L]
        w = x[L * N + L:]
        p_fin = er_rule_batch(r[rule_ids], w[rule_ids], beta[rule_ids])
        prob = p_fin[:, -1]
        prob = np.where(np.isfinite(prob), prob, 0.0)
        err = np.where(matched, (prob - y) ** 2, (1.0 - y) ** 2)
        return float(err.mean())

    return fitness


# ---------------- CMA-ES（对应 cmaes.m，含投影算子） ----------------
def _clip_and_project(arx, lb, ub, L, N):
    """先截断到 [lb,ub]，再对每条规则的置信度块做 sum=1 投影，最后负值置 0。
    与 cmaes.m 中投影算子逐行对应（含其'负值置0后不补偿'的实际行为）。"""
    np.clip(arx, lb, ub, out=arx)
    blocks = arx[:L * N].reshape(L, N, -1)                # (L,N,lambda)
    s = blocks.sum(axis=1, keepdims=True)
    blocks -= (s - 1.0) / N                               # 最小二乘投影到 sum=1
    blocks[blocks < 0] = 0.0
    return arx


def cmaes_min(x0, generations, fitness, L, N, seed=None, sigma0=0.5,
              lb=1e-4, ub=1.0, verbose=False):
    rng = np.random.default_rng(seed)
    n = len(x0)
    xmean = x0.astype(float).copy()
    sigma = sigma0

    lam = 10 + int(3 * np.log(n))                         # 后代数（N=560 → 28）
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = weights.sum() ** 2 / (weights ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

    pc = np.zeros(n); ps = np.zeros(n)
    B = np.eye(n); D = np.ones(n)
    C = np.eye(n); invsqrtC = np.eye(n)
    eigeneval = 0
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    counteval = 0
    bestf = np.inf; bestx = xmean.copy()
    while counteval < generations:
        arx = xmean[:, None] + sigma * (B @ (D[:, None] * rng.standard_normal((n, lam))))
        arx = _clip_and_project(arx, lb, ub, L, N)
        fits = np.array([fitness(arx[:, k]) for k in range(lam)])
        counteval += 1

        order = np.argsort(fits)
        if fits[order[0]] < bestf:
            bestf = fits[order[0]]
            bestx = arx[:, order[0]].copy()

        xold = xmean
        xmean = arx[:, order[:mu]] @ weights
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xold) / sigma
        hsig = (ps @ ps) / (1 - (1 - cs) ** (2 * counteval / lam)) / n < 2 + 4 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
        artmp = (arx[:, order[:mu]] - xold[:, None]) / sigma
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (artmp * weights) @ artmp.T)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        if counteval - eigeneval > lam / (c1 + cmu) / n / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T
            dh, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(dh, 1e-20))
            invsqrtC = B @ np.diag(1 / D) @ B.T
        if verbose and counteval % 50 == 0:
            print(f"  gen {counteval}: best={bestf:.6f} cur={fits[order[0]]:.6f} sigma={sigma:.3f}")
    return bestx, bestf


# ---------------- 顶层接口（接入 step1_pipeline） ----------------
def build_ibrb(X_train, y_train, method, cfg, seed=None, custom_boundaries=None):
    """
    method: "supervised"（信息增益划分）/ "equal_width"（等距划分）/ "mdlp"
    custom_boundaries: 预计算边界列表（消融变体 D），提供时忽略 method
    cfg 需含: ibrb.K_per_attr(=20), ibrb.mode, ibrb.cmaes_generations
    cfg["ibrb"]["optimize"]=False 时跳过 CMA-ES：Laplace 初始化 + r=0.9, w=1（消融变体 C）
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    M = X_train.shape[1]
    K = cfg["ibrb"].get("K_per_attr", 20)
    N = 2
    mode = cfg["ibrb"].get("mode", "faithful")
    optimize = cfg["ibrb"].get("optimize", True)

    if custom_boundaries is not None:
        bounds = [np.asarray(b, dtype=float) for b in custom_boundaries]
    elif method == "supervised":
        bounds = [info_gain_split(X_train[:, m], y_train, K, mode=mode,
                                  n_min=cfg["ibrb"].get("n_min", 5))
                  for m in range(M)]
    elif method == "equal_width":
        bounds = [equal_width_split(X_train[:, m], K) for m in range(M)]
    elif method == "mdlp":
        bounds = [mdlp_split(X_train[:, m], y_train, max_K=K) for m in range(M)]
    else:
        raise ValueError(method)
    L = int(sum(len(b) - 1 for b in bounds))

    if optimize:
        # 初始参数：对应 generator(2) —— 置信度交替 [1,0]/[0,1]，r、w 初值 0
        # （注意：与论文“Laplace 平滑/可靠度0.9”表述不符，见追踪表 R1-12）
        x0 = np.zeros(L * N + 2 * L)
        for k in range(L):
            x0[k * N:(k + 1) * N] = [1.0, 0.0] if k % 2 == 0 else [0.0, 1.0]

        fitness = make_fitness(X_train, y_train, bounds, L, N)
        bestx, bestf = cmaes_min(x0, cfg["ibrb"].get("cmaes_generations", 200),
                                 fitness, L, N, seed=seed,
                                 verbose=cfg["ibrb"].get("verbose", False))
        beta = bestx[:L * N].reshape(L, N)
        r = bestx[L * N:L * N + L]
        w = bestx[L * N + L:]
        model = IBRBModel(bounds, beta, r, w, n_grades=N)
        model.train_mse_ = bestf
    else:
        beta = laplace_init(X_train, y_train, bounds, n_grades=N)
        r = np.full(L, 0.9)
        w = np.ones(L)
        model = IBRBModel(bounds, beta, r, w, n_grades=N)
        model.train_mse_ = float(make_fitness(X_train, y_train, bounds, L, N)(
            np.concatenate([beta.ravel(), r, w])))
    return model


def predict_ibrb(model, X_test):
    prob, _ = predict_proba(model, X_test)
    return prob
