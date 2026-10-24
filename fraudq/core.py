"""core — 事件模型、截点可见性存储、16 项统计量与全局常量。

所有下游模块只依赖本文件的 Store.stats()/events()，保证"发生时间 vs 可用时间"
这一口径在全系统唯一实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

SEC, MIN, HOUR, DAY = 1_000, 60_000, 3_600_000, 86_400_000
CONNECT, UNANSWERED, BUSY, FAILED = 0, 1, 2, 3
STATUS_NAME = ("接通", "未接", "忙线", "其他失败")

REL_KINDS = ("D", "C", "I", "B", "P")
REL_NAME = {"D": "共享设备", "C": "共同目标", "I": "网络地址", "B": "基站位置", "P": "支付实体"}
# B（基站位置）改用 24 小时窗口：1 小时窗口内通话中位数为 0，"分母≥5"的质量门槛
# 会导致该路径在 618 个截点里只有 2 条可用，等于五条路径缺一条。
REL_WINDOW = {"D": 7 * DAY, "C": DAY, "I": DAY, "B": DAY, "P": 30 * DAY}
REL_GATE = {"D": .80, "C": .98, "I": .80, "B": .80, "P": .80}

STAT_NAMES = (
    "5分钟主叫数", "15分钟主叫数", "1小时主叫数", "24小时主叫数",
    "1小时被叫数", "1小时不同目标数", "新目标比例", "接通比例",
    "短接通比例", "平均接通时长", "双向联系比例", "最大目标重复数",
    "间隔变异系数", "夜间主叫比例", "30天活跃天数", "可观测历史长度",
)
# 需要 log(1+x) 变换的维度（计数与时长）
LOG_DIMS = np.array([0, 1, 2, 3, 4, 5, 9, 11, 14, 15])


@dataclass(slots=True)
class Call:
    cid: str
    caller: int
    callee: int
    start: int
    end: int
    avail: int
    talk: int
    status: int
    dur_valid: bool
    caller_ok: bool
    callee_ok: bool
    batch: int = 0


@dataclass(slots=True)
class Obs:
    oid: str
    user: int
    kind: str
    key: str
    at: int
    avail: int
    valid: bool = True
    etype: str = "account"      # 仅支付实体使用：account / merchant / terminal


@dataclass(slots=True)
class Query:
    qid: str
    user: int
    cutoff: int
    label: int = -1
    q_call: float = 1.0
    q_d: float = 1.0
    identity: bool = True


@dataclass
class Sample:
    """一次截点查询的冻结输入。"""
    q: Query
    x: np.ndarray            # 32 维画像（16 标准化 + 16 掩码）
    m: np.ndarray            # 16 掩码
    last_event: np.ndarray   # 12 维最近一条事件向量
    n_ev: int                # 参与聚合的可见事件数
    history_days: float


class Store:
    """按截点 t 过滤的可见记录仓库；同时提供窗口统计与关系候选。"""

    def __init__(self, calls: Sequence[Call], obs: Sequence[Obs], stats: Optional[dict] = None):
        self.calls = list(calls)
        self.obs = list(obs)
        self.stat_mean = np.zeros(16) if stats is None else np.asarray(stats["med"], float)
        self.stat_iqr = np.ones(16) if stats is None else np.asarray(stats["iqr"], float)
        self._build()

    # ---------- 构建索引 ----------
    def _build(self) -> None:
        self.E: Dict[int, np.ndarray] = {}
        self.EC: Dict[int, List[Call]] = {}
        self.EID: Dict[int, List[str]] = {}
        self.CU: Dict[int, List[Call]] = {}
        self._cids = set()
        self._oids = set()
        for c in self.calls:
            self._cids.add(c.cid)
            self.CU.setdefault(c.caller, []).append(c)
            self.CU.setdefault(c.callee, []).append(c)
        for u in self.CU:
            self._rebuild_arrays(u)

        self.R: Dict[str, Dict[str, List[tuple]]] = {k: {} for k in REL_KINDS}
        self.RU: Dict[str, Dict[int, List[tuple]]] = {k: {} for k in REL_KINDS}
        # C 路径(共同被叫目标)直接由通话记录生成，不依赖外部观测
        for c in self.calls:
            self.R["C"].setdefault(str(c.callee), []).append((c.caller, c.start, c.avail))
            self.RU["C"].setdefault(c.caller, []).append((str(c.callee), c.start, c.avail))
        for o in self.obs:
            if o.valid:
                self.R[o.kind].setdefault(o.key, []).append((o.user, o.at, o.avail))
                self.RU[o.kind].setdefault(o.user, []).append((o.key, o.at, o.avail))

    def _rebuild_arrays(self, u: int) -> None:
        rows = []
        for c in self.CU.get(u, ()):
            role = 1 if c.caller == u else 0
            peer = c.callee if role else c.caller
            rows.append((c.avail, c.start, role, c.status, c.talk, peer, c))
        rows.sort(key=lambda r: (r[0], r[1]))
        self.E[u] = (np.array([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=np.int64)
                     if rows else np.zeros((0, 6), np.int64))
        self.EC[u] = [r[6] for r in rows]
        self.EID[u] = [r[6].cid for r in rows]

    # ---------- 增量写入 ----------
    def add_call(self, c: Call) -> bool:
        """在线插入一条通话；同 call_id 幂等。返回是否真的写入。"""
        if c.cid in self._cids:
            return False
        self._cids.add(c.cid)
        self.calls.append(c)
        for u in {c.caller, c.callee}:
            self.CU.setdefault(u, []).append(c)
            self._rebuild_arrays(u)
        self.R["C"].setdefault(str(c.callee), []).append((c.caller, c.start, c.avail))
        self.RU["C"].setdefault(c.caller, []).append((str(c.callee), c.start, c.avail))
        return True

    def add_obs(self, o: Obs) -> bool:
        if o.oid in self._oids:
            return False
        self._oids.add(o.oid)
        self.obs.append(o)
        if o.valid:
            self.R[o.kind].setdefault(o.key, []).append((o.user, o.at, o.avail))
            self.RU[o.kind].setdefault(o.user, []).append((o.key, o.at, o.avail))
        return True

    def latest(self, u: int) -> int:
        a = self.E.get(u)
        return int(a[:, 0].max()) if a is not None and len(a) else 0

    # ---------- 可见性 ----------
    def vis_mask(self, u: int, t: int, skip: frozenset = frozenset()) -> np.ndarray:
        a = self.E.get(u)
        if a is None:
            return np.zeros(0, bool)
        m = a[:, 0] <= t
        if skip:
            ids = self.EID.get(u, [])
            m &= np.array([cid not in skip for cid in ids], bool)
        return m

    def visible(self, u: int, t: int, skip: frozenset = frozenset()) -> np.ndarray:
        a = self.E.get(u)
        return np.zeros((0, 6), np.int64) if a is None else a[self.vis_mask(u, t, skip)]

    def history_days(self, u: int, t: int) -> float:
        a = self.E.get(u)
        if a is None or len(a) == 0:
            return 0.0
        first = a[:, 1].min()
        return float(min(30.0, max(0.0, (t - first) / DAY)))

    # ---------- 16 项统计量 ----------
    def raw_stats(self, u: int, t: int, skip: frozenset = frozenset()) -> tuple[np.ndarray, np.ndarray]:
        a = self.E.get(u)
        val = np.zeros(16)
        msk = np.zeros(16)
        if a is None or len(a) == 0:
            return val, msk
        av, st, role, status, talk, peer = (a[:, i] for i in range(6))
        vis = self.vis_mask(u, t, skip)
        if not vis.any():
            return val, msk
        st, role, status, talk, peer = st[vis], role[vis], status[vis], talk[vis], peer[vis]
        cal = role == 1
        con = status == CONNECT

        def cnt(mask: np.ndarray) -> int:
            return int(mask.sum())

        for i, w in ((0, 5 * MIN), (1, 15 * MIN), (2, HOUR), (3, DAY)):
            wm = st >= t - w
            val[i] = cnt(wm & cal)
        w1h = st >= t - HOUR
        val[4] = cnt(w1h & ~cal)
        tgt = peer[w1h & cal]
        val[5] = len(np.unique(tgt))
        # 新目标比例：1h 目标中此前 30d~1h 未出现者
        old = peer[(st >= t - 30 * DAY) & (st < t - HOUR) & cal]
        val[6] = float((~np.isin(tgt, old)).mean()) if len(tgt) else 0.0
        msk[6] = float(len(tgt) > 0 and (t - st.min()) >= 30 * DAY)
        att = cnt(w1h & cal)
        val[7] = cnt(w1h & cal & con) / att if att else 0.0
        msk[7] = float(att > 0)
        dv = w1h & con & (talk > 0)
        val[8] = float((talk[dv] <= 10).mean()) if dv.any() else 0.0
        msk[8] = float(dv.any())
        val[9] = float(talk[dv].mean()) if dv.any() else 0.0
        msk[9] = float(dv.any())
        inc = peer[(st >= t - DAY) & ~cal]
        val[10] = float(np.isin(tgt, inc).mean()) if len(tgt) else 0.0
        msk[10] = float(len(tgt) > 0)
        if len(tgt):
            _, c = np.unique(tgt, return_counts=True)
            val[11] = float(c.max())
        msk[11] = float(len(tgt) > 0)
        ts = np.sort(st[w1h & cal])
        if len(ts) >= 3:
            d = np.diff(ts)
            val[12] = float(d.std() / d.mean()) if d.mean() > 0 else 0.0
            msk[12] = 1.0
        w24 = st >= t - DAY
        hh = ((st[w24 & cal] // HOUR) % 24)
        val[13] = float((hh < 6).mean()) if len(hh) else 0.0
        msk[13] = float(len(hh) > 0)
        w30 = st >= t - 30 * DAY
        val[14] = float(len(np.unique(st[w30] // DAY)))
        msk[14] = float((t - st.min()) >= 30 * DAY)
        val[15] = float(min(30.0, max(0.0, (t - st.min()) / DAY)))
        msk[15] = 1.0
        msk[0:6] = 1.0
        return val, msk

    def profile(self, u: int, t: int, skip: frozenset = frozenset()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (32 维画像, 16 掩码, 标准化后 16 值)。"""
        raw, msk = self.raw_stats(u, t, skip)
        g = raw.copy()
        g[LOG_DIMS] = np.log1p(np.maximum(raw[LOG_DIMS], 0))
        z = (g - self.stat_mean) / np.maximum(self.stat_iqr, 1e-6)
        z = np.clip(z, -5, 5)
        z[msk == 0] = 0.0
        x = np.concatenate([z, msk])
        return x, msk, z

    # ---------- 12 维事件向量 ----------
    def event_vector(self, c: Call, u: int, t: int) -> np.ndarray:
        role = 1.0 if c.caller == u else 0.0
        talk = c.talk if c.dur_valid else 0
        e = np.zeros(12)
        e[0] = np.log1p(max(talk, 0))
        e[1 + c.status] = 1.0
        h = (c.start // HOUR) % 24 + (c.start % HOUR) / HOUR
        e[5] = np.sin(2 * np.pi * h / 24)
        e[6] = np.cos(2 * np.pi * h / 24)
        e[7] = np.log1p(max(c.end - c.start, 0))
        e[8] = np.log1p(max(c.avail - c.end, 0))
        e[9] = 1.0 if c.dur_valid else 0.0
        e[10] = 1.0 if (c.callee_ok if role else c.caller_ok) else 0.0
        e[11] = role
        return e

    # ---------- 快通道：10 分钟爆发信号 ----------
    FAST_N, FAST_NEW = 14, .80

    def fast_signal(self, u: int, t: int, win: int = 10 * MIN):
        """返回 (10 分钟内主叫数, 新目标比例, 是否触发)。纯规则，<1ms。"""
        a = self.E.get(u)
        if a is None or len(a) == 0:
            return 0, 0.0, False
        m = self.vis_mask(u, t)
        st, role, peer = a[m, 1], a[m, 2], a[m, 5]
        w = (st >= t - win) & (role == 1)
        c = int(w.sum())
        tgt = peer[w]
        if c == 0:
            return 0, 0.0, False
        old = peer[(st >= t - 30 * DAY) & (st < t - win) & (role == 1)]
        newr = float((~np.isin(tgt, old)).mean()) if len(old) else 1.0
        return c, newr, bool(c >= self.FAST_N and newr >= self.FAST_NEW)

    # ---------- 关系候选 ----------
    def relations(self, u: int, t: int, kind: str, skip_keys: frozenset = frozenset()):
        """返回 (本端关联键, {键: 成员列表}, {键: 度数})。"""
        win = REL_WINDOW[kind]
        keys: Dict[str, int] = {}
        for key, at, av in self.RU[kind].get(u, ()):
            if av <= t and at >= t - win and key not in skip_keys:
                keys[key] = 1
        if not keys:
            return {}, {}, {}
        deg, members = {}, {}
        for key in keys:
            seen = set()
            for v, at, av in self.R[kind][key]:
                if av <= t and at >= t - win:
                    seen.add(v)
            members[key] = list(seen)
            deg[key] = len(seen)
        return keys, members, deg

    def relation_weight(self, u: int, t: int, kind: str,
                        skip_keys: frozenset = frozenset()) -> Dict[int, float]:
        keys, members, deg = self.relations(u, t, kind, skip_keys)
        w: Dict[int, float] = {}
        for key in keys:
            for v in members[key]:
                if v == u:
                    continue
                w[v] = w.get(v, 0.0) + 1.0 / np.log(2.0 + max(deg[key], 1))
        if not w:
            return {}
        s = float(np.sqrt(sum(x * x for x in w.values())) + 1e-8)
        return {v: x / s for v, x in w.items()}

    # ---------- 训练统计量 ----------
    def fit_stats(self, users: Sequence[int], t: int) -> dict:
        rows = [self.raw_stats(u, t)[0] for u in users]
        arr = np.array(rows)
        g = arr.copy()
        g[:, LOG_DIMS] = np.log1p(np.maximum(arr[:, LOG_DIMS], 0))
        med = np.median(g, axis=0)
        iqr = np.subtract(*np.percentile(g, [75, 25], axis=0))
        iqr[iqr < 1e-6] = 1.0
        self.stat_mean, self.stat_iqr = med, iqr
        return {"med": med.tolist(), "iqr": iqr.tolist()}

# ============================ 运行时上下文 ============================
import time as _time


class Budget:
    """时延预算：不足时降级，不编造。"""

    def __init__(self, total_ms: float = 200.0):
        self.total = float(total_ms)
        self.left = float(total_ms)
        self.spent = 0.0

    def afford(self, ms: float) -> bool:
        return self.left >= ms

    def spend(self, ms: float) -> None:
        self.left -= ms
        self.spent += ms


@dataclass
class Trace:
    qid: str
    user: int
    cutoff: int
    results: Dict[str, dict] = field(default_factory=dict)
    decisions: List[dict] = field(default_factory=list)
    status: str = ""
    action: str = ""
    elapsed_ms: float = 0.0
    audit: dict = field(default_factory=dict)

    def log(self, rule: str, cond: str, choice: str, cost: float, reason: str = "") -> None:
        self.decisions.append({"rule": rule, "cond": cond, "choice": choice,
                               "cost_ms": round(float(cost), 2), "reason": reason})


@dataclass
class Context:
    q: Query
    store: "Store"
    sample: "Sample"
    budget: Budget
    trace: Trace
    engine: object
    skip_calls: frozenset = frozenset()
    skip_keys: dict = field(default_factory=dict)
    sink: object = None
    _t0: float = field(default_factory=_time.perf_counter)

    def emit(self, stage: str, msg: str, **kw) -> None:
        """把执行过程旁白推给前端（CLI 的 sink），None 时静默。"""
        if self.sink is not None:
            self.sink(stage, msg, kw)

    def put(self, key: str, **kw) -> None:
        self.trace.results[key] = kw

    def get(self, key: str, default=None) -> dict:
        if key in self.trace.results:
            return self.trace.results[key]
        return {} if default is None else default

    def has(self, key: str) -> bool:
        return key in self.trace.results

    def reset_clock(self) -> None:
        self._t0 = _time.perf_counter()

    def elapsed(self) -> float:
        return (_time.perf_counter() - self._t0) * 1000.0
