"""s2 — 时序模式核验智能体：事件率反比双尺度窗口 + EWMA/CUSUM + 漂移 + 双模式库。

按专利说明书 2.1–2.4 对齐：
  * EWMA 初值取窗口前一个桶（与 5 分钟桶同量纲），不再用 1 小时画像；
  * 稳健离散尺度 s_i 取用户自身最近 7 天相邻桶一阶差分 MAD（带相对下限与兜底）；
  * 比例类统计量按桶内分母判定有效性，无效桶不参与 CUSUM 与打分；
  * CUSUM 判定阈值按同时检验的 16 维校正（h=6.5），避免噪声维每 12 个桶假报一次；
  * 长时漂移的 s_ref 取参考基线原始日级序列 MAD（平滑只用于均值），并带下限；
  * 长时五项按表 3 对应统计量取权重；
  * 12 维模式片段按表 3 重排（含设备/基站标识序列）；
  * 异常模式库按爆发型/渐进型分层，并执行 200/40 最低样本量门槛。
"""
from __future__ import annotations

import numpy as np

from .core import DAY, HOUR, MIN, REL_KINDS

SHORT_W = (5 * MIN, 2 * HOUR)
LONG_W = (7 * DAY, 30 * DAY)
T_B = 5 * MIN
BETA_S, BETA_D = .30, .20
K_R = 1.0
H_TH = 6.5
A_S, B_S_ = .8, -.5
A_L, B_L_ = .6, -.8
KAPPA_V, THETA_L = .25, 2.0
L_REF = L_REC = 14 * DAY
TAU_ALIGN = .456 * L_REC
C0, TH_BS, TH_BL = .30, .50, .50
W_TS, ETA0, ALPHA, GAMMA = .5, .5, .6, .5
SCALE_DAYS = 7
S_Q = 0.75
# 逐维尺度下限（比例类 0.1；秒数类 5；计数类 1）
FLOOR_LONG = np.array([1,1,1,1,1,1, .1,.1,.1,5., .1,1., .1,.1, 1.,1.], float)
FLOOR_SHORT = np.array([.5,.5,.5,.5,.5,.5, .1,.1,.1,5., .1,.5, .1,.1, .5,.5], float)
MIN_BANK, MIN_ABN = 200, 40
ROLL_D = 3

# 表 3 长时五项对应的表 1 统计量（0 基）：24h主叫、不同目标、新目标、双向联系、夜间、活跃
U_DIMS = [3, 5, 6, 10, 13, 14]


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


def bucket_span(t: int, K: int, T_B: int, a: int, b: int) -> tuple[int, int]:
    """第 a..b 个桶覆盖的时间区间 (start, end)，单位毫秒。

    桶 k（0 基）覆盖 [t-(K-k)·T_B, t-(K-k-1)·T_B]。返回值恒满足 start <= end、
    end-start = (b-a+1)·T_B —— 端点一旦写反，时间一致性会被 max(0, ·) 压成 0，
    「双尺度互证」就永远不可能触发。
    """
    return t - (K - a) * T_B, t - (K - b - 1) * T_B


def _cos(p: np.ndarray, M: np.ndarray) -> np.ndarray:
    p = p / (np.linalg.norm(p) + 1e-9)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M @ p


class S2Agent:
    name = "S2"

    def __init__(self, norm_bank=None, abn_bank=None, med=None, iqr=None,
                 bank_ok: bool = True):
        self.norm: np.ndarray = np.zeros((1, 12)) if norm_bank is None else norm_bank
        self.abn: np.ndarray = np.zeros((1, 12)) if abn_bank is None else abn_bank
        self.med = np.zeros(12) if med is None else med
        self.iqr = np.ones(12) if iqr is None else iqr
        self.bank_ok = bank_ok

    def estimate(self, ctx) -> float:
        return 1.2 * max(ctx.engine.cost_hint("S2"), 4.0)

    def eligible(self, ctx) -> bool:
        return ctx.sample.history_days >= 7.0

    def skip(self, ctx, reason: str) -> None:
        t = ctx.q.cutoff
        ctx.put("S2", B_s=0.0, B_l=0.0, C_temp=0.0, lam=0.0, R=0.0, E_ts=0.0,
                flag="未执行", reason=reason, W_s=float("nan"), W_l=float("nan"),
                K=0, D=0, t_s=(t, t), t_l=(t, t), series=np.zeros(1))

    # ---------- 窗口 ----------
    def windows(self, ctx) -> tuple[float, float]:
        from .ablate import CFG
        if CFG.get("fixed_window"):
            return float(2 * HOUR), float(14 * DAY)
        st, t, u = ctx.store, ctx.q.cutoff, ctx.q.user
        a = st.visible(u, t)
        n24 = float(((a[:, 1] >= t - DAY)).sum()) if len(a) else 0.0
        n30 = float(((a[:, 1] >= t - 30 * DAY)).sum()) if len(a) else 0.0
        rs = n24 / 86400.0          # 条/秒
        rd = n30 / 30.0             # 条/天
        ws = float(np.clip(3 / max(rs, 1e-6) * 1000.0, *SHORT_W))
        wl = float(np.clip(20 / max(rd, 1e-6) * DAY, *LONG_W))
        return ws, wl

    # ---------- 逐通话派生量 ----------
    def _flags(self, a: np.ndarray) -> dict:
        n = len(a)
        start, role, status, talk, peer = a[:, 1], a[:, 2].astype(int), a[:, 3], a[:, 4], a[:, 5]
        cal = role == 1
        con = (status == 0) & (talk > 0)
        is_new = np.zeros(n, bool)
        is_bidir = np.zeros(n, bool)
        last_out, last_in = {}, {}
        order = np.argsort(start, kind="mergesort")
        for i in order:
            s = int(start[i]); p = int(peer[i])
            if cal[i]:
                prev = last_out.get(p)
                is_new[i] = not (prev is not None and HOUR <= s - prev <= 30 * DAY)
                last_out[p] = s
                prev_in = last_in.get(p)
                is_bidir[i] = prev_in is not None and s - prev_in <= DAY
            else:
                last_in[p] = s
        return {"cal": cal, "inc": ~cal, "con": con,
                "short": con & (talk <= 10), "night": ((start // HOUR) % 24) < 6,
                "is_new": is_new, "is_bidir": is_bidir}

    # ---------- 桶内 16 项统计量与逐维有效性 ----------
    def _bucket(self, a: np.ndarray, f: dict, t0: int, t1: int):
        x = np.zeros(16)
        v = np.zeros(16, bool)
        x[15] = (t1 - t0) / DAY
        v[0:6] = True
        v[15] = True
        if len(a) == 0:
            return x, v
        start, peer = a[:, 1], a[:, 5]
        lo = int(np.searchsorted(start, t0, side="left"))
        hi = int(np.searchsorted(start, t1, side="left"))
        if hi <= lo:
            return x, v
        cal = f["cal"][lo:hi]
        tgt = peer[lo:hi][cal]
        ncal, ninc = int(cal.sum()), int((~cal).sum())
        x[0] = x[1] = x[2] = x[3] = ncal
        x[4] = ninc
        x[5] = len(np.unique(tgt)) if len(tgt) else 0.0
        if len(tgt):
            _, cnt = np.unique(tgt, return_counts=True)
            x[6] = float(f["is_new"][lo:hi][cal].mean())
            x[10] = float(f["is_bidir"][lo:hi][cal].mean())
            x[11] = float(cnt.max())
        con = cal & f["con"][lo:hi]
        x[7] = con.sum() / max(ncal, 1)
        if con.any():
            x[8] = float(f["short"][lo:hi][con].mean())
            x[9] = float(a[lo:hi, 4][con].mean())
        ts = start[lo:hi][cal]
        if len(ts) >= 3:
            d = np.diff(np.sort(ts))
            x[12] = float(d.std() / d.mean()) if d.mean() > 0 else 0.0
        if ncal:
            x[13] = float(f["night"][lo:hi][cal].mean())
        x[14] = float(len(np.unique(start[lo:hi] // DAY)))
        v[6] = len(tgt) > 0
        v[7] = ncal >= 3
        v[8] = v[9] = bool(con.any())
        v[10] = len(tgt) > 0
        v[11] = len(tgt) > 0
        v[12] = len(ts) >= 3
        v[13] = ncal > 0
        v[14] = True
        return x, v

    # ---------- 整段序列（向量化，供 7 天尺度与日级窗口使用）----------
    def _series(self, a: np.ndarray, f: dict, edges: np.ndarray):
        """一次算完整段 16×nb 统计量与有效性，语义与 _bucket 相同。"""
        nb = len(edges) - 1
        B = np.zeros((nb, 16)); V = np.zeros((nb, 16), bool)
        B[:, 15] = np.diff(edges) / DAY
        V[:, 0:6] = True; V[:, 15] = True
        if len(a) == 0 or nb <= 0:
            return B, V
        start, peer, talk = a[:, 1], a[:, 5], a[:, 4]
        b = np.searchsorted(edges, start, side="right") - 1
        ok = (b >= 0) & (b < nb)
        if not ok.any():
            return B, V
        b = b[ok]; start = start[ok]
        cal = f["cal"][ok]; con = f["con"][ok]; short = f["short"][ok]
        night = f["night"][ok]; isnew = f["is_new"][ok]; isbid = f["is_bidir"][ok]
        pr = peer[ok]; tl = talk[ok]
        any_cnt = np.bincount(b, minlength=nb)
        B[:, 14] = (any_cnt > 0).astype(float)
        V[:, 14] = any_cnt > 0
        cc = np.bincount(b[cal], minlength=nb).astype(float)
        B[:, 0] = B[:, 1] = B[:, 2] = B[:, 3] = cc
        B[:, 4] = np.bincount(b[~cal], minlength=nb)
        if cal.any():
            bt = b[cal]; pt = pr[cal]
            big = int(pt.max()) + 1
            uk, cnts = np.unique(bt.astype(np.int64) * big + pt, return_counts=True)
            ub = (uk // big).astype(int)
            B[:, 5] = np.bincount(ub, minlength=nb)
            mx = np.zeros(nb); np.maximum.at(mx, ub, cnts); B[:, 11] = mx
            n_new = np.bincount(bt, weights=isnew[cal], minlength=nb)
            n_bid = np.bincount(bt, weights=isbid[cal], minlength=nb)
            n_con = np.bincount(bt, weights=con[cal], minlength=nb)
            n_short = np.bincount(bt, weights=(con & short)[cal], minlength=nb)
            n_talk = np.bincount(bt, weights=tl[cal] * con[cal], minlength=nb)
            n_night = np.bincount(bt, weights=night[cal], minlength=nb)
            B[:, 6] = n_new / np.maximum(cc, 1)
            B[:, 10] = n_bid / np.maximum(cc, 1)
            B[:, 7] = n_con / np.maximum(cc, 1)
            B[:, 8] = n_short / np.maximum(n_con, 1)
            B[:, 9] = n_talk / np.maximum(n_con, 1)
            B[:, 13] = n_night / np.maximum(cc, 1)
            V[:, 6] = B[:, 5] > 0
            V[:, 7] = cc >= 3
            V[:, 8] = V[:, 9] = n_con > 0
            V[:, 10] = B[:, 5] > 0
            V[:, 11] = B[:, 5] > 0
            V[:, 13] = cc > 0
            for k in np.flatnonzero(cc >= 3):
                ts = np.sort(start[(b == k) & cal])
                d = np.diff(ts)
                if len(d) and d.mean() > 0:
                    B[k, 12] = d.std() / d.mean()
                    V[k, 12] = True
        return B, V

    # ---------- 稳健离散尺度 ----------
    def _scale(self, ctx, a: np.ndarray, f: dict) -> np.ndarray:
        eng = ctx.engine
        cache = getattr(eng, "_s2_scale", None)
        if cache is None:
            cache = eng._s2_scale = {}
        key = (ctx.q.user, ctx.q.cutoff)
        if key in cache:
            return cache[key]
        t = ctx.q.cutoff
        nb = int(SCALE_DAYS * DAY // T_B)
        B, V = self._series(a, f, np.arange(t - nb * T_B, t + 1, T_B))
        s = np.ones(16)
        for d in range(16):
            xs = B[V[:, d], d]
            lvl = float(np.mean(np.abs(xs))) if len(xs) else float(np.mean(np.abs(B[:, d])))
            if len(xs) >= 6:
                mad = float(np.quantile(np.abs(np.diff(xs)), S_Q))
            else:
                mad = 0.0
            s[d] = max(mad, np.sqrt(max(lvl, 0.0)), 0.1 * lvl, float(FLOOR_SHORT[d]))
        cache[key] = s
        return s

    # ---------- 主流程 ----------
    def run(self, ctx) -> None:
        t, u = ctx.q.cutoff, ctx.q.user
        ws, wl = self.windows(ctx)
        K = max(1, int(ws // T_B))
        D = max(1, int(wl // DAY))
        # 桶统计用 searchsorted 按下界切段，必须先按事件发生时间排序
        # （visible 是按"可用时间"返回的，延迟到达会造成局部逆序）
        a = ctx.store.visible(u, t, ctx.skip_calls)
        if len(a):
            a = a[np.argsort(a[:, 1], kind="stable")]
        f = self._flags(a)

        # ---- 短时：逐桶 EWMA + 双侧 CUSUM ----
        X = np.zeros((K, 16)); V = np.zeros((K, 16), bool)
        for k in range(K):
            X[k], V[k] = self._bucket(a, f, t - (K - k) * T_B, t - (K - k - 1) * T_B)
        m_prev, _ = self._bucket(a, f, t - (K + 1) * T_B, t - K * T_B)
        s_i = self._scale(ctx, a, f)
        e = np.zeros((K, 16))
        Cp = np.zeros(16); Cn = np.zeros(16)
        hit = np.zeros((K, 16), bool)
        for k in range(K):
            ok = V[k]
            e[k] = np.where(ok, (X[k] - m_prev) / (s_i + 1e-6), 0.0)
            m_prev = np.where(ok, BETA_S * X[k] + (1 - BETA_S) * m_prev, m_prev)
            for i in range(16):
                if not ok[i]:
                    continue
                Cp[i] = max(0.0, Cp[i] + e[k, i] - K_R)
                Cn[i] = max(0.0, Cn[i] - e[k, i] - K_R)
                if Cp[i] > H_TH or Cn[i] > H_TH:
                    hit[k, i] = True
                    Cp[i] = Cn[i] = 0.0
        w = np.array([1.0] * 6 + [.5] * 10)
        any_hit = hit.any(1)
        per_bucket = np.where(any_hit, A_S * (w[None, :] * np.abs(e) * hit).max(1) + B_S_, B_S_)
        bs_k = 1 / (1 + np.exp(-per_bucket))
        B_s = float(bs_k.max()) if K else 0.0
        t_s_start, t_s_end = t, t
        spans = []
        for i in range(16):
            idx = np.flatnonzero(hit[:, i])
            if not len(idx):
                continue
            best = [idx[0], idx[0]]; cur = [idx[0], idx[0]]
            for k in idx[1:]:
                if k - cur[1] <= 2:
                    cur[1] = k
                else:
                    if cur[1] - cur[0] > best[1] - best[0]:
                        best = list(cur)
                    cur = [k, k]
            if cur[1] - cur[0] > best[1] - best[0]:
                best = list(cur)
            spans.append(bucket_span(t, K, T_B, best[0], best[1]))
        if spans:
            t_s_start = min(x[0] for x in spans)
            t_s_end = max(x[1] for x in spans)

        # ---- 长时：固定参考基线（截点前 30~16 天）与近期窗口（前 14 天）----
        ref_start = t - 30 * DAY
        rec_start = t - L_REC
        xr, _ = self._series(a, f, np.arange(ref_start, ref_start + 15 * DAY, DAY))
        xg, _ = self._series(a, f, np.arange(rec_start, rec_start + 15 * DAY, DAY))

        def smooth(A):
            out = np.zeros_like(A); prev = A[0]
            for j in range(len(A)):
                prev = BETA_D * A[j] + (1 - BETA_D) * prev
                out[j] = prev
            return out

        sr, sg = smooth(xr), smooth(xg)
        mu_ref, mu_rec = sr.mean(0), sg.mean(0)
        med_ref = np.median(xr, axis=0)
        lvl_ref = np.mean(np.abs(xr), axis=0)
        s_ref = np.median(np.abs(xr - med_ref), axis=0)
        s_ref = np.maximum(s_ref, np.sqrt(np.maximum(lvl_ref, 0.0)))
        s_ref = np.maximum(s_ref, 0.1 * lvl_ref)
        s_ref = np.maximum(s_ref, FLOOR_LONG)
        g = np.abs(mu_rec - mu_ref) / (s_ref + 1e-6)
        jj = np.arange(1, 15); jb = jj.mean()
        v = ((jj - jb)[:, None] * (sg - mu_rec)).sum(0) / ((jj - jb) ** 2).sum()
        u_i = np.zeros(16); u_i[U_DIMS] = 1.0
        B_l = float(1 / (1 + np.exp(-(A_L * (u_i * (g + KAPPA_V * np.maximum(0, v * (L_REC / DAY)) /
                                                    (s_ref + 1e-6))).max() + B_L_))))
        roll = np.stack([sg[max(0, i - ROLL_D + 1):i + 1].mean(0) for i in range(14)])
        g3 = np.abs(roll - mu_ref) / (s_ref + 1e-6)
        hot = (u_i[None, :] > 0) & (g3 > THETA_L)
        t_l_start, t_l_end = t, t
        spans = []
        for i in range(16):
            idx = np.flatnonzero(hot[:, i])
            if not len(idx):
                continue
            best = [idx[0], idx[0]]; cur = [idx[0], idx[0]]
            for k in idx[1:]:
                if k - cur[1] <= 2:
                    cur[1] = k
                else:
                    if cur[1] - cur[0] > best[1] - best[0]:
                        best = list(cur)
                    cur = [k, k]
            if cur[1] - cur[0] > best[1] - best[0]:
                best = list(cur)
            spans.append((rec_start + best[0] * DAY, rec_start + (best[1] + 1) * DAY))
        if spans:
            t_l_start = min(x[0] for x in spans)
            t_l_end = max(x[1] for x in spans)

        # ---- 12 维模式片段与双库匹配 ----
        p = self._pattern(ctx, X, e, V, t, K, ws)
        pz = (p - self.med) / np.maximum(self.iqr, 1e-6)
        if self.bank_ok:
            R = min(1.0, max(0.0, float(_cos(pz, self.abn).max() - _cos(pz, self.norm).max())))
        else:
            R = 0.0

        # ---- 时间一致性 + 证据权重 ----
        rs_ = max(0.0, (t_s_end - t_s_start) / ws)
        rl_ = max(0.0, (t_l_end - t_l_start) / L_REC)
        gap = max(0.0, (t_s_start - t_l_end) / 1000.0)
        c_temp = float(np.sqrt(min(1, rs_) * min(1, rl_)) * np.exp(-gap / (TAU_ALIGN / 1000)))
        if B_s >= TH_BS and B_l >= TH_BL and c_temp >= C0:
            lam, flag = 1.4, "已互证"          # 双尺度互证：提权
        elif B_s >= TH_BS and B_l >= TH_BL:
            lam, flag = .65, "未互证"          # 两尺度都越阈但时间对不上：降权
        elif B_s >= TH_BS or B_l >= TH_BL:
            lam, flag = .65, "待核验"          # 单一窗口异常：降权
        else:
            lam, flag = 0.0, "无核验证据"
        e_ts = W_TS * lam * (ETA0 + (1 - ETA0) * c_temp) * (ALPHA * B_s + (1 - ALPHA) * B_l) * (1 + GAMMA * R)
        ctx.put("S2", B_s=B_s, B_l=B_l, C_temp=c_temp, lam=lam, R=R, E_ts=float(e_ts), pat=p,
                flag=flag, W_s=ws, W_l=wl, K=K, D=D, s_i=s_i,
                t_s=(t_s_start, t_s_end), t_l=(t_l_start, t_l_end), series=bs_k)
        ctx.trace.log("S2", f"hist>={7:.0f}d", flag, ctx.elapsed(),
                      f"W_s={ws/60000:.0f}min W_l={wl/DAY:.0f}d B_s={B_s:.2f} B_l={B_l:.2f} "
                      f"C_temp={c_temp:.2f} λ={lam} E_ts={e_ts:.3f}")

    # ---------- 12 维模式 ----------
    def _key_sets(self, ctx, kind: str, edges: np.ndarray) -> list:
        out = [set() for _ in range(len(edges) - 1)]
        for key, at, av in ctx.store.RU.get(kind, {}).get(ctx.q.user, ()):
            if av > ctx.q.cutoff:
                continue
            i = int(np.searchsorted(edges, at, side="right") - 1)
            if 0 <= i < len(out):
                out[i].add(key)
        return out

    def _pattern(self, ctx, X, e, V, t, K, ws):
        p = np.zeros(12)
        if K < 2:
            return p
        first, last = X[0], X[-1]
        p[0] = np.log((last[0] + 1) / (first[0] + 1))
        p[1] = last[5] - first[5]
        p[2] = last[6] - first[6]
        edges = np.array([t - (K - k) * T_B for k in range(K + 1)])
        dev = self._key_sets(ctx, "D", edges)
        bs = self._key_sets(ctx, "B", edges)
        p[3] = float(sum(1 for k in range(1, K) if dev[k] != dev[k - 1]))
        p[4] = float(np.mean([bs[k] != bs[k - 1] for k in range(1, K)]))
        p[5] = last[13] - first[13]
        p[6] = last[10] - first[10]
        p[7] = last[8] - first[8]
        ratio = X[:, 0] / np.maximum(X[:, 4], 1e-9)
        p[8] = float((np.diff(np.sign(ratio)) != 0).sum())
        p[9] = last[12] - first[12]
        p[10] = np.log((last[9] + 1) / (first[9] + 1))
        ev = np.abs(e)[V]
        p[11] = float(np.quantile(ev, .9)) if len(ev) else 0.0
        return p

    # ---------- 训练：建双模式库 ----------
    def fit(self, feats: np.ndarray, labels: np.ndarray,
            b_s: np.ndarray = None, b_l: np.ndarray = None) -> None:
        self.med = np.median(feats, 0)
        q = np.subtract(*np.percentile(feats, [75, 25], axis=0))
        self.iqr = np.where(q < 1e-6, 1.0, q)
        z = (feats - self.med) / self.iqr
        pos = labels == 0
        if pos.sum() >= 16:
            self.norm = _kmeans(z[pos], 16, seed=1)
        else:
            self.norm = z[pos] if pos.any() else z[:1]
        abn = z[labels == 1]
        if len(abn):
            if b_s is not None and b_l is not None:
                bs1 = b_s[labels == 1]; bl1 = b_l[labels == 1]
                r = bs1 / (bs1 + bl1 + 1e-9)
                split = r >= np.median(r)
            else:
                split = np.arange(len(abn)) < max(1, len(abn) // 2)
            parts = [abn[split], abn[~split]]
            banks = [_kmeans(part, 4, seed=2 + i) if len(part) else z[:1]
                     for i, part in enumerate(parts)]
            self.abn = np.vstack(banks)
        else:
            self.abn = z[:1]
        self.bank_ok = bool((pos.sum() >= MIN_BANK) and (len(abn) >= MIN_ABN))
