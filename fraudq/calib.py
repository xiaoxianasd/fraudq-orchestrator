"""calib — 置信度检验：可靠性图、ECE、Brier/NLL、温度缩放、风险-覆盖、置信指标分层。"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

EPS = 1e-6


def collect(coord, queries, budget: float = 200.0, route_all: bool = True,
            replay_mode: str = "defer") -> List[dict]:
    """route_all=True 时强制全路径，测的是「评分函数本身」的置信度；
    默认 False 时测的是含门控/降级后的「系统输出」置信度。"""
    out = []
    for q in queries:
        tr = coord.run(q, budget, route_all=route_all, replay_mode=replay_mode)
        r = tr.results
        f, s1, s2, han, sc = (r.get("FUSE", {}), r.get("S1", {}), r.get("S2", {}),
                              r.get("HAN", {}), r.get("SCREEN", {}))
        out.append({
            "qid": q.qid, "user": q.user, "cutoff": q.cutoff, "label": int(q.label),
            "p": float(f.get("p", sc.get("p0", 0.0))),
            "logit": float(f.get("logit", 0.0)),
            "p_T": float(s1.get("p_cal", np.nan)), "raw_T": float(s1.get("p", np.nan)),
            "c_sem": float(s1.get("c_sem", 0.0)),
            "p_H": float(han.get("p_cal", np.nan)),
            "E_ts": float(s2.get("E_ts", 0.0)), "C_temp": float(s2.get("C_temp", 0.0)),
            "lam": float(s2.get("lam", 0.0)), "flag": s2.get("flag", "未执行"),
            "B_s": float(s2.get("B_s", 0.0)), "B_l": float(s2.get("B_l", 0.0)),
            "p0": float(sc.get("p0", 0.0)), "u": float(sc.get("u", 1.0)),
            "g_u": int(sc.get("g", 0)), "status": tr.status, "ms": float(tr.elapsed_ms),
            "has_s2": bool(s2), "has_han": bool(han),
        })
    return out


def _bins(p: np.ndarray, n: int = 10, quantile: bool = True):
    """等量分箱（按预测值分位），避免正例稀少时个别箱样本过少导致 ECE 失真。"""
    p = np.asarray(p, float)
    if not quantile:
        return np.minimum((p * n).astype(int), n - 1)
    qs = np.quantile(p, np.linspace(0, 1, n + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    return np.clip(np.searchsorted(qs, p, side="right") - 1, 0, n - 1)


def ece(y: np.ndarray, p: np.ndarray, n: int = 10) -> float:
    idx = _bins(p, n)
    tot = 0.0
    for b in range(n):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def mce(y: np.ndarray, p: np.ndarray, n: int = 10) -> float:
    idx = _bins(p, n)
    worst = 0.0
    for b in range(n):
        m = idx == b
        if m.sum() < 3:
            continue
        worst = max(worst, abs(y[m].mean() - p[m].mean()))
    return float(worst)


def brier(y, p) -> float:
    return float(np.mean((p - y) ** 2))


def nll(y, p) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auroc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean())


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def temperature(y, p, t: float) -> np.ndarray:
    z = _logit(p) / t
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(y, p, grid=None) -> float:
    """在给定数据上最小化 NLL 求温度 T（保序，只改锐度）。"""
    if grid is None:
        grid = np.concatenate([np.linspace(.2, 1.0, 33), np.linspace(1.1, 6.0, 50)])
    best, bt = 1e9, 1.0
    for t in grid:
        v = nll(y, temperature(y, p, t))
        if v < best:
            best, bt = v, float(t)
    return bt


def cv_temperature(y, p, k: int = 5, seed: int = 0) -> tuple[np.ndarray, List[float]]:
    """K 折交叉拟合温度，返回折外校准后的概率与各折 T。"""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.random.default_rng(seed).permutation(len(y))
    out = np.zeros_like(p)
    ts = []
    for f in range(k):
        te = idx[f::k]
        tr = np.setdiff1d(idx, te)
        t = fit_temperature(y[tr], p[tr])
        ts.append(t)
        out[te] = temperature(y[te], p[te], t)
    return out, ts


def summary(y, p) -> Dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"n": int(len(y)), "pos": int(y.sum()), "acc@0.5": float(((p >= .5) == y).mean()),
            "auroc": auroc(y, p), "brier": brier(y, p), "nll": nll(y, p),
            "ece": ece(y, p), "mce": mce(y, p), "mean_p": float(p.mean()), "mean_y": float(y.mean())}


def stratify(y, p, key, nq: int = 4) -> List[dict]:
    """按某个置信度指标四分位分层，报告各层样本量、准确率与 ECE。"""
    y, p, key = np.asarray(y, float), np.asarray(p, float), np.asarray(key, float)
    qs = np.quantile(key, np.linspace(0, 1, nq + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    rows = []
    for i in range(nq):
        m = (key > qs[i]) & (key <= qs[i + 1])
        if m.sum() == 0:
            continue
        rows.append({"lo": float(qs[i] if np.isfinite(qs[i]) else key.min()),
                     "hi": float(qs[i + 1] if np.isfinite(qs[i + 1]) else key.max()),
                     "n": int(m.sum()), "pos": int(y[m].sum()),
                     "acc": float(((p[m] >= .5) == y[m]).mean()),
                     "ece": ece(y[m], p[m], 6), "mean_p": float(p[m].mean()),
                     "mean_y": float(y[m].mean())})
    return rows


def risk_coverage(y, p) -> Dict[str, np.ndarray]:
    """按分数从高到低放行：覆盖率 vs 被放行样本的精确率/召回率。"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(-p)
    ys = y[order]
    k = np.arange(1, len(ys) + 1)
    tp = np.cumsum(ys)
    prec = tp / k
    rec = tp / max(ys.sum(), 1)
    cov = k / len(ys)
    return {"coverage": cov, "precision": prec, "recall": rec,
            "threshold": p[order]}


def save(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)