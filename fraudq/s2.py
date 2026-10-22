"""s2 — 时序模式核验智能体：事件率反比双尺度窗口 + EWMA/CUSUM + 漂移 + 双模式库。"""
from __future__ import annotations

import numpy as np

from .core import DAY, HOUR, MIN, REL_KINDS

SHORT_W = (5 * MIN, 2 * HOUR)
LONG_W = (7 * DAY, 30 * DAY)
T_B = 5 * MIN
BETA_S, BETA_D = .30, .20
K_R, H_TH = 1.0, 5.0
A_S, B_S_ = .8, -.5
A_L, B_L_ = .6, -.8
KAPPA_V, THETA_L = .25, 2.0
L_REF = L_REC = 14 * DAY
TAU_ALIGN = .456 * L_REC
C0, TH_BS, TH_BL = .30, .50, .50
W_TS, ETA0, ALPHA, GAMMA = .5, .5, .6, .5


def _kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    if len(X) <= k:
        return X.copy()
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return C


def _cos(p: np.ndarray, M: np.ndarray) -> np.ndarray:
    p = p / (np.linalg.norm(p) + 1e-9)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M @ p


class S2Agent:
    name = "S2"

    def __init__(self, norm_bank=None, abn_bank=None, med=None, iqr=None):
        self.norm: np.ndarray = np.zeros((1, 12)) if norm_bank is None else norm_bank
        self.abn: np.ndarray = np.zeros((1, 12)) if abn_bank is None else abn_bank
        self.med = np.zeros(12) if med is None else med
        self.iqr = np.ones(12) if iqr is None else iqr

    def estimate(self, ctx) -> float:
        return 4.0 + 0.1 * ctx.engine.cost_hint("S2")

    def eligible(self, ctx) -> bool:
        return ctx.sample.history_days >= 7.0

    # ---------- 窗口 ----------
    def windows(self, ctx) -> tuple[float, float]:
        st, t, u = ctx.store, ctx.q.cutoff, ctx.q.user
        a = st.visible(u, t)
        n24 = float(((a[:, 1] >= t - DAY)).sum()) if len(a) else 0.0
        n30 = float(((a[:, 1] >= t - 30 * DAY)).sum()) if len(a) else 0.0
        rs = n24 / 86400.0          # 条/秒
        rd = n30 / 30.0             # 条/天
        ws = float(np.clip(3 / max(rs, 1e-6) * 1000.0, *SHORT_W))
        wl = float(np.clip(20 / max(rd, 1e-6) * DAY, *LONG_W))
        return ws, wl

    def _bucket_stats(self, ctx, starts: np.ndarray, t0: int, t1: int) -> np.ndarray:
        """在 [t0,t1) 内的 16 维统计（复用 core.raw_stats 的窗口语义）。"""
        out = np.zeros(16)
        a = ctx.store.E.get(ctx.q.user)
        if a is None:
            return out
        vis = ctx.store.vis_mask(ctx.q.user, ctx.q.cutoff, ctx.skip_calls) & (a[:, 1] >= t0) & (a[:, 1] < t1)
        if not vis.any():
            return out
        st, role, status, talk, peer = a[vis, 1], a[vis, 2], a[vis, 3], a[vis, 4], a[vis, 5]
        cal = role == 1
        con = status == 0
        out[0] = out[1] = out[2] = out[3] = cal.sum()
        out[4] = (~cal).sum()
        tgt = peer[cal]
        out[5] = len(np.unique(tgt))
        out[6] = 1.0 - len(np.unique(tgt)) / max(len(tgt), 1)
        out[7] = (cal & con).sum() / max(cal.sum(), 1)
        dv = con & (talk > 0)
        out[8] = float((talk[dv] <= 10).mean()) if dv.any() else 0.0
        out[9] = float(talk[dv].mean()) if dv.any() else 0.0
        out[10] = 1.0 - len(np.unique(tgt)) / max(len(tgt), 1)
        if len(tgt):
            _, c = np.unique(tgt, return_counts=True)
            out[11] = c.max()
        ts = np.sort(st[cal])
        if len(ts) > 2:
            d = np.diff(ts)
            out[12] = d.std() / d.mean() if d.mean() > 0 else 0.0
        hh = (st[(st >= t1 - DAY) & cal] // HOUR) % 24
        out[13] = float((hh < 6).mean()) if len(hh) else 0.0
        out[14] = len(np.unique(st // DAY))
        out[15] = (t1 - t0) / DAY
        return out

    def run(self, ctx) -> None:
        t, u = ctx.q.cutoff, ctx.q.user
        ws, wl = self.windows(ctx)
        K = max(1, int(ws // T_B))
        D = max(1, int(wl // DAY))
        # ---- 短时：逐桶 EWMA + 双侧 CUSUM ----
        X = np.stack([self._bucket_stats(ctx, None, t - (K - k) * T_B, t - (K - k - 1) * T_B)
                      for k in range(K)])
        raw = ctx.store.raw_stats(u, t, ctx.skip_calls)[0]
        m0 = raw
        s_i = np.maximum(np.array([1.0] * 16), 1e-3)
        e = np.zeros_like(X)
        Cp = np.zeros(16)
        Cn = np.zeros(16)
        anomaly = np.zeros(K, dtype=bool)
        m_prev = m0.copy()
        for k in range(K):
            e[k] = (X[k] - m_prev) / (s_i + 1e-6)
            m_prev = BETA_S * X[k] + (1 - BETA_S) * m_prev
            hit = np.zeros(16, dtype=bool)
            for i in range(16):
                Cp[i] = max(0.0, Cp[i] + e[k, i] - K_R)
                Cn[i] = max(0.0, Cn[i] - e[k, i] - K_R)
                if Cp[i] > H_TH or Cn[i] > H_TH:
                    hit[i] = True
                    Cp[i] = Cn[i] = 0.0
            anomaly[k] = hit.any()
        w = np.array([1.0] * 6 + [.5] * 10)
        bs_k = 1 / (1 + np.exp(-(A_S * (w[None, :] * np.abs(e) * anomaly[:, None]).max(1) + B_S_)))
        B_s = float(bs_k.max())
        ts_idx = np.flatnonzero(anomaly)
        t_s_start = t - (K - ts_idx.min()) * T_B if len(ts_idx) else t
        t_s_end = t - (K - ts_idx.max() - 1) * T_B if len(ts_idx) else t

        # ---- 长时：固定参考基线与近期窗口 ----
        ref_end, ref_start = t - 17 * DAY, t - 31 * DAY
        rec_start = t - L_REC
        x_ref = np.array([self._bucket_stats(ctx, None, ref_start + j * DAY, ref_start + (j + 1) * DAY)
                          for j in range(14)])
        x_rec = np.array([self._bucket_stats(ctx, None, rec_start + j * DAY, rec_start + (j + 1) * DAY)
                          for j in range(14)])
        def smooth(A):
            out = np.zeros_like(A); prev = A[0]
            for j in range(len(A)):
                prev = BETA_D * A[j] + (1 - BETA_D) * prev; out[j] = prev
            return out
        sr, sg = smooth(x_ref), smooth(x_rec)
        mu_ref, mu_rec = sr.mean(0), sg.mean(0)
        s_ref = np.median(np.abs(np.diff(sr, axis=0)), axis=0)
        s_ref = np.maximum(s_ref, 1e-3)
        g_i = np.abs(mu_rec - mu_ref) / (s_ref + 1e-6)
        j = np.arange(1, 15); jb = j.mean()
        v_i = ((j - jb)[:, None] * (sg - mu_rec)).sum(0) / ((j - jb) ** 2).sum()
        u_i = np.zeros(16); u_i[[6, 5, 13, 9, 14]] = 1.0
        B_l = float(1 / (1 + np.exp(-(A_L * (u_i * (g_i + KAPPA_V * np.maximum(0, v_i * 14) /
               (s_ref + 1e-6))).max() + B_L_))))
        dev = (u_i[None, :] + 1e-9) * np.abs(sg - mu_ref) / (s_ref + 1e-6)
        hit = np.flatnonzero(dev.max(1) > THETA_L)
        t_l_start = rec_start + int(hit.min() + 1) * DAY if len(hit) else t
        t_l_end = rec_start + int(hit.max() + 2) * DAY if len(hit) else t

        # ---- 12 维模式片段与双库匹配 ----
        p = self._pattern(ctx, X)
        pz = (p - self.med) / np.maximum(self.iqr, 1e-6)
        R = max(0.0, float(_cos(pz, self.abn).max() - _cos(pz, self.norm).max()))

        # ---- 时间一致性 + 证据权重 ----
        rs_ = max(0.0, (t_s_end - t_s_start) / ws)
        rl_ = max(0.0, (t_l_end - t_l_start) / L_REC)
        gap = max(0.0, (t_s_start - t_l_end) / 1000.0)
        c_temp = float(np.sqrt(min(1, rs_) * min(1, rl_)) * np.exp(-gap / (TAU_ALIGN / 1000)))
        if B_s >= TH_BS and B_l >= TH_BL and c_temp >= C0:
            lam, flag = 1.4, "已互证"
        elif B_s >= TH_BS or B_l >= TH_BL:
            lam, flag = .65, "待核验"
        else:
            lam, flag = 0.0, "无核验证据"
        e_ts = W_TS * lam * (ETA0 + (1 - ETA0) * c_temp) * (ALPHA * B_s + (1 - ALPHA) * B_l) * (1 + GAMMA * R)
        ctx.put("S2", B_s=B_s, B_l=B_l, C_temp=c_temp, lam=lam, R=R, E_ts=float(e_ts), pat=p,
                flag=flag, W_s=ws, W_l=wl, K=K, D=D,
                t_s=(t_s_start, t_s_end), t_l=(t_l_start, t_l_end), series=bs_k)
        ctx.trace.log("S2", f"hist>={7:.0f}d", flag, ctx.elapsed(),
                      f"W_s={ws/60000:.0f}min W_l={wl/DAY:.0f}d B_s={B_s:.2f} B_l={B_l:.2f} "
                      f"C_temp={c_temp:.2f} λ={lam} E_ts={e_ts:.3f}")

    def _pattern(self, ctx, X: np.ndarray) -> np.ndarray:
        p = np.zeros(12)
        if len(X) < 2:
            return p
        c = X[:, [0, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
        first, last = c[0], c[-1]
        p[0] = np.log((last[0] + 1) / (first[0] + 1))
        p[1] = last[1] - first[1]
        p[2] = last[2] - first[2]
        p[3] = float((np.diff(X[:, 5]) != 0).sum())
        p[4] = float((np.diff(X[:, 6]) != 0).mean())
        p[5] = last[5] - first[5]
        p[6] = last[6] - first[6]
        p[7] = last[7] - first[7]
        p[8] = float((np.diff(np.sign(last[0] - first[0]) * np.ones(len(X))) != 0).sum())
        p[9] = last[8] - first[8]
        p[10] = np.log((last[9] + 1) / (first[9] + 1))
        p[11] = float(np.quantile(np.abs(X[:, 0]), .9))
        return p

    # ---------- 训练：建双模式库 ----------
    def fit(self, feats: np.ndarray, labels: np.ndarray) -> None:
        self.med = np.median(feats, 0)
        q = np.subtract(*np.percentile(feats, [75, 25], axis=0))
        self.iqr = np.where(q < 1e-6, 1.0, q)
        z = (feats - self.med) / self.iqr
        self.norm = _kmeans(z[labels == 0], 16, seed=1) if (labels == 0).sum() > 0 else z[:1]
        abn = z[labels == 1]
        if len(abn):
            half = max(1, len(abn) // 2)
            self.abn = np.vstack([_kmeans(abn[:half], 4, seed=2), _kmeans(abn[half:], 4, seed=3)])
        else:
            self.abn = z[:1]
