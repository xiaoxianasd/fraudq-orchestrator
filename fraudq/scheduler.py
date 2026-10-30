"""scheduler — 多智能体协同调度器（本发明的核心创新落点）。

调度不是固定 DAG，而是"证据驱动的动态任务插入"：
    frontier 在每次 agent.run() 之后按新证据重算，可解锁 HAN / REPLAY。
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .ablate import CFG
from .core import (DAY, HOUR, MIN, REL_GATE, REL_KINDS, Budget, Context, Query, Sample,
                   Store, Trace)
from .replay import ReplayAgent
from .s1 import S1Agent, TGN
from .s2 import S2Agent
from .s3 import Fusion, FusionAgent, HAN, Platt, S3HanAgent

RELEASE, ESCALATE, FULL = "release", "escalate", "full"

PEER_HORIZON = 3 * DAY     # 对端记忆的固定回放窗口


class Screener(nn.Module):
    """44→16→(:均值, :对数方差)，恰 754 个参数。"""

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(44, 16)
        self.mu = nn.Linear(16, 1)
        self.lv = nn.Linear(16, 1)

    def forward(self, x):
        h = torch.relu(self.body(x))
        return self.mu(h).squeeze(-1), self.lv(h).squeeze(-1).clamp(-8, 8)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Engine:
    def __init__(self, store: Store, models: dict, peer_mode: str = "global"):
        self.store = store
        self.m = models
        self.peer_mode = peer_mode
        self._peer: Dict[int, Dict[int, torch.Tensor]] = {}
        self._prof: Dict[tuple, np.ndarray] = {}
        self._cost: Dict[str, float] = {}
        self._h: Dict[tuple, torch.Tensor] = {}
        self.version = models.get("version", "v1")

    # ---------- 缓存 ----------
    def profile(self, u: int, t: int) -> np.ndarray:
        k = (u, t)
        if k not in self._prof:
            self._prof[k] = self.store.profile(u, t)[0]
        return self._prof[k]

    DEFAULT_COST = {"S1": 45.0, "S2": 10.0, "HAN": 15.0, "FUSE": 1.0, "REPLAY": 200.0}

    def cost_hint(self, name: str) -> float:
        return self._cost.get(name, self.DEFAULT_COST.get(name, 5.0))

    def note_cost(self, name: str, ms: float) -> None:
        old = self.cost_hint(name)
        ms = min(ms, max(old * 6.0, 50.0), 150.0)   # 抑制预热尖峰；绝对上限 150ms
        self._cost[name] = 0.7 * old + 0.3 * ms

    def invalidate(self, from_ts: int) -> None:
        """新记录写入后：清画像/邻居缓存，丢弃受影响截点的记忆快照。"""
        self._prof.clear()
        self._h.clear()
        self._W = None                     # TGN 权重的 NumPy 副本也要一并失效
        for t in [k for k in self._peer if k >= from_ts]:
            self._peer.pop(t, None)

    # ---------- 全局记忆快照 ----------
    def peer_memory(self, t: int) -> Dict[int, torch.Tensor]:
        if t in self._peer:
            return self._peer[t]
        mem: Dict[int, torch.Tensor] = {}
        if self.peer_mode == "none":
            self._peer[t] = mem
            return mem
        W = self._np_weights()
        te0 = np.zeros(16, np.float32); te0[1::2] = 1.0        # TE(0) = [0,1,0,1,...]
        zero = np.zeros(128, np.float32)
        # 固定回放窗口：同一截点的记忆只由 [t-H, t] 的记录决定，与"之前算过哪些截点"无关。
        # 不做增量续算——否则结果会依赖缓存历史，破坏"同一截点可复现"。
        mem, lo = {}, t - PEER_HORIZON
        calls = [c for c in self.store.calls if lo <= c.avail <= t]
        calls.sort(key=lambda c: (c.avail, c.start))
        for c in calls:
            a, b = c.caller, c.callee
            sa, sb = mem.get(a, zero), mem.get(b, zero)
            ea = self.store.event_vector(c, a, t)
            eb = self.store.event_vector(c, b, t)
            g = 1.0 if (c.status == 0 and c.talk * 1000 >= 5000) else 0.0
            na = _np_step(np.concatenate([sa, sb]), te0, ea, W)
            nb = _np_step(np.concatenate([sb, g * sa]), te0, eb, W)
            mem[a], mem[b] = na, nb
        self._peer[t] = {u: torch.from_numpy(v) for u, v in mem.items()}
        return self._peer[t]

    def _np_weights(self) -> dict:
        if getattr(self, "_W", None) is not None:
            return self._W
        tgn = self.m["tgn"]

        def lin(l):
            return l.weight.detach().numpy().T.astype(np.float32), (
                None if l.bias is None else l.bias.detach().numpy().astype(np.float32))
        W = {"m1": lin(tgn.msg[0]), "m2": lin(tgn.msg[2]), "z": lin(tgn.wz), "uz": lin(tgn.uz),
             "r": lin(tgn.wr), "ur": lin(tgn.ur), "h": lin(tgn.wh), "uh": lin(tgn.uh)}
        self._W = W
        return W

    def peer_h(self, v: int, t: int) -> torch.Tensor:
        k = (v, t)
        if k not in self._h:
            s = self.peer_memory(t).get(v, torch.zeros(128))
            x = torch.tensor(self.profile(v, t), dtype=torch.float32)
            with torch.no_grad():
                self._h[k] = torch.relu(self.m["tgn"].base(torch.cat([s, x])))
        return self._h[k]

    def platt(self, kind: str, logit: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.m["platt_" + kind](logit)

    # ---------- 样本 ----------
    def prepare(self, q: Query, skip: frozenset = frozenset()) -> Sample:
        x, m, _ = self.store.profile(q.user, q.cutoff, skip)
        a = self.store.visible(q.user, q.cutoff, skip)
        if len(a):
            idx = int(np.flatnonzero(self.store.vis_mask(q.user, q.cutoff, skip))[-1])
            last = self.store.event_vector(self.store.EC[q.user][idx], q.user, q.cutoff)
        else:
            last = np.zeros(12)
        return Sample(q, x, m, last, len(a), self.store.history_days(q.user, q.cutoff))

    # ---------- 隔离重放：带屏蔽重算 ----------
    def rescore(self, ctx: Context, skip_calls: frozenset, skip_keys: dict) -> float:
        """屏蔽证据后按原路径重算，返回综合风险分数的概率值。

        与首评的差别只有三点：画像/记忆/统计按 skip 重建；只重跑首评实际执行过的分支
        （不把"HAN 原本没跑"变成"跑了"）；其余参数与版本一律固定。
        """
        sub = Context(ctx.q, self.store, self.prepare(ctx.q, frozenset(skip_calls)),
                      Budget(1e9), Trace(ctx.q.qid, ctx.q.user, ctx.q.cutoff), self)
        sub.skip_calls = frozenset(skip_calls)
        sub.skip_keys = skip_keys
        relq = ctx.get("RELQ")
        if relq:
            sub.put("RELQ", **relq)
        ran = set(ctx.trace.results.keys())
        with torch.no_grad():
            for ag in self.branches:
                if ag.name in ran and ag.eligible(sub):
                    ag.run(sub)
        logit = float(sub.get("FUSE").get("logit", 0.0))
        return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit)))))


class Coordinator:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.s1 = S1Agent(engine.m["tgn"])
        self.s2 = S2Agent(engine.m["s2_norm"], engine.m["s2_abn"], engine.m["s2_med"],
                          engine.m["s2_iqr"], bank_ok=bool(engine.m.get("s2_ok", True)))
        self.han = S3HanAgent(engine.m["han"])
        self.fuse = FusionAgent(engine.m["fusion"])
        self.replay = ReplayAgent()
        engine.branches = [self.s1, self.s2, self.han, self.fuse]

    # ---------- 触发条件 ----------
    @staticmethod
    def relation_trigger(ctx) -> tuple[bool, str]:
        s1 = ctx.get("S1")
        u, t = ctx.q.user, ctx.q.cutoff
        raw, _ = ctx.store.raw_stats(u, t)
        if s1 and s1["p"] >= .20:
            return True, "行为侧 p_T>=0.20"
        if raw[1] >= 30 and raw[5] >= 25 and raw[6] >= .80:
            return True, "行为侧 15min>=30 & 目标>=25 & 新目标>=0.8"
        return False, ""

    def run(self, q: Query, budget_ms: float = 200.0, route_all: bool = False,
            replay_mode: str = "sync", sink=None) -> Trace:
        eng = self.engine
        tr = Trace(q.qid, q.user, q.cutoff)
        ctx = Context(q, eng.store, eng.prepare(q), Budget(budget_ms), tr, eng)
        ctx.sink = sink
        tr._ctx = ctx
        st = eng.store

        # ---- 0 加载可见数据 ----
        vis = st.visible(q.user, q.cutoff)
        rels = {k: len(st.relation_weight(q.user, q.cutoff, k)) for k in REL_KINDS}
        ctx.emit("data", "可见记录加载完成",
                 visible=len(vis), mask=int(ctx.sample.m.sum()),
                 rels=rels, history_days=round(ctx.sample.history_days, 1),
                 cutoff=q.cutoff)

        # ---- 0.5 快通道：10 分钟爆发规则 ----
        c10, newr, fast = st.fast_signal(q.user, q.cutoff)
        ctx.put("FAST", c10=c10, new_ratio=newr, hit=fast)
        tr.log("FAST", f"10min主叫={c10} 新目标={newr:.2f}",
               "命中快通道" if fast else "未命中", 0.2, "阈值 14 通 / 0.80")
        ctx.emit("fast", "10 分钟爆发规则", calls=c10, new_ratio=round(newr, 3), hit=fast,
                 rule=f">= {st.FAST_N} 通 且 新目标 >= {st.FAST_NEW:.2f}")

        # 数据质量分级：完整性不足不等于不可判，只封顶动作
        dq = "ok" if (q.identity and q.q_call >= .98) else \
             ("degraded" if (q.identity and q.q_call >= .90) else "bad")
        ctx.put("DQ", level=dq, q_call=q.q_call)
        if dq != "ok":
            ctx.emit("dq", "数据质量降级", level=dq, q_call=round(q.q_call, 3),
                     ceiling="禁止阻断，最高预警" if dq == "degraded" else "不予判断")
            tr.log("DQ", f"q_call={q.q_call:.3f} identity={q.identity}", dq,
                   0.0, "0.90~0.98 降级判断；<0.90 或身份未知才判信息不足")
        if dq == "bad":
            return self._finish(ctx, "信息不足", "无",
                                f"话单完整性 {q.q_call:.3f} <0.90 或身份不可用")

        # ---- 1 轻筛（754 参数）----
        p0, u, g_u = self._screen(ctx)
        ctx.put("SCREEN", p0=p0, u=u, g=g_u)
        tr.log("SCREEN", f"mask={int(ctx.sample.m.sum())}", f"p0={p0:.3f} u={u:.3f} G_u={int(g_u)}",
               ctx.elapsed(), "44 维轻量模型")
        ctx.emit("screen", "轻量初筛 754 参数", p0=round(p0, 4), u=round(u, 4), g_u=bool(g_u))

        # ---- 2 一级判别：S1 ----
        self._run(ctx, self.s1)
        s1 = ctx.get("S1", {})
        p_t = s1.get("p", 0.0)
        if not CFG.get("all_branches") and g_u and p0 < .20 and ctx.has("S1") and p_t < .15 and not route_all and not fast:
            tr.log("GATE", f"p0={p0:.3f}<0.20 & p_T={p_t:.3f}<0.15 & G_u=1",
                   "级联放行", ctx.elapsed(), "两级都低才放行")
            ctx.emit("gate", "两级判别均为低风险，放行", p0=round(p0, 4), p_t=round(p_t, 4),
                     next="终止")
            return self._finish(ctx, "低危", "放行", "两级判别一致且均低风险")
        if fast and g_u and p0 < .20:
            ctx.emit("gate", "快通道命中，禁止放行", next="继续")
        if g_u and p0 >= .85:
            tr.log("GATE", "p0>=0.85 & G_u=1", "高风险升级", 0.0, "跳过低价值分支，直取关系证据")
            ctx.emit("gate", "初筛即高置信高风险", next="升级")
        else:
            tr.log("GATE", "0.20<=p0<0.85 或 G_u=0", "并行取证 S1+S2", 0.0, "边界区/高不确定度")

        # ---- 3 时序核验 ----
        self._run(ctx, self.s2)

        # ---- 4 按需关系核查 ----
        trig, why = self.relation_trigger(ctx)
        quals, cands, avail = self.han.available(ctx)
        # 数据质量与路径可用标记属于数据本身，与 HAN 是否执行无关：
        # 单独落一份给融合层，避免 HAN 未触发时被当作"质量=0/路径全不可用"
        ctx.put("RELQ", quals=quals, avail=avail)
        if (trig or CFG.get("all_branches")) and any(avail.values()):
            tr.log("TRIGGER", why + f" & Σm_φ={sum(avail.values())}", "插入 HAN", 0.0, "关系侧条件成立")
            ctx.emit("trigger", "关系侧条件成立，插入 HAN", reason=why,
                     paths=[k for k in REL_KINDS if avail[k]])
            self._run(ctx, self.han)
        else:
            tr.log("TRIGGER", why or "无关系触发", "跳过 HAN", 0.0,
                   "无可用路径" if trig else "触发条件未满足")
            ctx.emit("trigger", "跳过关系核查", reason=why or "未满足触发条件",
                     avail=[k for k in REL_KINDS if avail[k]])

        # ---- 5 融合（唯一不可降级的分支：融合缺失会把"有证据"变成"零风险"）----
        self._run(ctx, self.fuse, force=True)
        f = ctx.get("FUSE", {})
        p = f.get("p", 0.0)
        gap = ReplayAgent.gap(ctx)
        ctx.emit("fuse", "15 维可信度融合", p=round(p, 4), gap=round(gap, 3))

        # ---- 6 复核（同步/异步）----
        if (p >= .80 or gap >= .30) and not CFG.get("no_replay"):
            if replay_mode == "defer":
                tr.log("TRIGGER", f"p={p:.3f} gap={gap:.3f}", "挂起异步复核", 0.0,
                       "重放只加证据不改分，移出在线路径")
                ctx.put("REPLAY", pending=True, groups=self.replay.groups(ctx))
                ctx.emit("replay", "触发条件成立，转入异步复核", p=round(p, 4),
                         gap=round(gap, 3), mode="async")
            else:
                tr.log("TRIGGER", f"p={p:.3f} gap={gap:.3f}", "插入隔离重放", 0.0, "高分或分支冲突")
                ctx.emit("replay", "触发条件成立，在线隔离重放", p=round(p, 4), gap=round(gap, 3),
                         mode="sync")
                self._run(ctx, self.replay)
        else:
            tr.log("TRIGGER", f"p={p:.3f} gap={gap:.3f}", "跳过重放", 0.0, "未达触发阈值")
            ctx.emit("replay", "未达复核阈值，跳过", p=round(p, 4), gap=round(gap, 3))

        return self._finish(ctx, "", "", "")

    def finalize(self, trace: Trace, budget_ms: float = 2000.0) -> tuple[Trace, float]:
        """异步复核：补跑挂起的隔离重放并重算状态。返回 (trace, 复核耗时ms)。"""
        ctx = getattr(trace, "_ctx", None)
        if ctx is None or not ctx.get("REPLAY").get("pending"):
            return trace, 0.0
        ctx.budget = Budget(budget_ms)
        t0 = time.perf_counter()
        with torch.no_grad():
            self.replay.run(ctx)
        ms = (time.perf_counter() - t0) * 1000.0
        trace.status, trace.action = Judge.decide(ctx)
        trace.log("JUDGE", trace.status, trace.action, ms, "异步复核回填")
        return trace, ms

    # ---------- 工具 ----------
    def _screen(self, ctx) -> tuple[float, float, bool]:
        if ctx.sample.m.sum() == 0:
            return 1.0, 1.0, False
        x = np.concatenate([ctx.sample.x, ctx.sample.last_event])
        with torch.no_grad():
            mu, lv = self.engine.m["screener"](torch.tensor(x, dtype=torch.float32))
            p0 = float(torch.sigmoid(mu))
            u = float(min(1.0, math.exp(float(lv))))
        if ctx.sample.history_days < 7:
            u = 1.0
        return p0, u, (u <= .25)

    def _run(self, ctx, agent, force: bool = False) -> None:
        eng = self.engine
        est = agent.estimate(ctx)
        if not agent.eligible(ctx):
            ctx.trace.log(agent.name, "eligible=false", "跳过", 0.0, "条件不满足")
            ctx.emit("skip", "跳过", agent=agent.name, reason="执行条件不满足")
            if hasattr(agent, "skip"):
                agent.skip(ctx, "执行条件不满足")
            else:
                ctx.put(agent.name, skipped=True, reason="执行条件不满足")
            return
        if not force and not ctx.budget.afford(est):
            ctx.trace.log(agent.name, f"预算剩余={ctx.budget.left:.1f}ms < {est:.1f}ms",
                          "降级跳过", 0.0, "以掩码位进入融合，不按零风险处理")
            ctx.emit("skip", "预算不足，降级跳过", agent=agent.name,
                     reason=f"剩余 {ctx.budget.left:.1f}ms < 预估 {est:.1f}ms")
            if hasattr(agent, "skip"):
                agent.skip(ctx, "剩余时延预算不足")
            else:
                ctx.put(agent.name, skipped=True, reason="剩余时延预算不足")
            ctx.trace.results[agent.name]["budget"] = True
            return
        ctx.reset_clock()
        with torch.no_grad():
            agent.run(ctx)
        ms = ctx.elapsed()
        ctx.budget.spend(ms)
        eng.note_cost(agent.name, ms)
        self._narrate(ctx, agent, ms)

    @staticmethod
    def _verdict(p: float) -> str:
        """分支分数的展示口径，与表6 一致（0.80 / 0.20）。"""
        return "高危" if p >= .80 else ("可疑" if p >= .20 else "低危")

    def _narrate(self, ctx, agent, ms: float) -> None:
        n = agent.name
        if n == "S1":
            d = ctx.get("S1", {})
            ctx.emit("S1", "行为语义智能体",
                     p_T=round(d.get("p", 0), 4), p_cal=round(d.get("p_cal", 0), 4),
                     c_sem=round(d.get("c_sem", 0), 3), checkpoints=d.get("checkpoints", 0),
                     verdict=self._verdict(d.get("p", 0)), ms=round(ms, 1),
                     next="转时序核验")
        elif n == "S2":
            d = ctx.get("S2", {})
            ctx.emit("S2", "时序核验 " + d.get("flag", "未执行"),
                     B_s=round(d.get("B_s", 0), 3), B_l=round(d.get("B_l", 0), 3),
                     C_temp=round(d.get("C_temp", 0), 3), lam=d.get("lam", 0),
                     E_ts=round(d.get("E_ts", 0), 4),
                     ws=(None if d.get("flag") == "未执行" else round(d.get("W_s", 0) / 60000)),
                      wl=(None if d.get("flag") == "未执行" else round(d.get("W_l", 0) / 86400000)),
                     ms=round(ms, 1), next="转关系核查")
        elif n == "HAN":
            d = ctx.get("HAN", {})
            beta = d.get("beta") or [0] * 5
            paths = {k: round(b, 2) for k, b in zip(REL_KINDS, beta) if b > 1e-3}
            ctx.emit("HAN", "五元路径关系核查", p_H=round(d.get("p_cal", 0), 4),
                     p_raw=round(d.get("p", 0), 4),
                     paths=paths, avail=[k for k, v in (d.get("avail") or {}).items() if v],
                     verdict=self._verdict(d.get("p", 0)), ms=round(ms, 1), next="转融合")
        elif n == "REPLAY":
            d = ctx.get("REPLAY", {})
            sc = d.get("scenes", {})
            ctx.emit("REPLAY", "隔离重放完成",
                     scenes={k: v["delta"] for k, v in sc.items()}, best=round(d.get("best", 0), 3),
                     ms=round(ms, 1))

    def _finish(self, ctx, status: str, action: str, reason: str) -> Trace:
        ctx.trace.elapsed_ms = ctx.budget.spent
        if not status:
            status, action = Judge.decide(ctx)
        ctx.trace.status, ctx.trace.action = status, action
        ctx.trace.log("JUDGE", status, action, 0.0, reason)
        f = ctx.get("FUSE", {})
        p = f.get("p", 0.0)
        ctx.emit("judge", f"{status} → {action}", action=action, p=round(p, 4),
                 p_lo=round(f.get("p_lo", 0), 4), p_hi=round(f.get("p_hi", 0), 4),
                 verdict=self._verdict(p) if ctx.has("FUSE") else "—",
                 ms=round(ctx.budget.spent, 1))
        # 处置动作 + 事后审计附件（人工只用于阻断/预警之后，不在在线路径上）
        ctx.trace.audit = {"grade": status, "action": action,
                           "p": round(p, 4), "p_lo": round(f.get("p_lo", 0), 4),
                           "p_hi": round(f.get("p_hi", 0), 4),
                           "evidence_top": (ctx.get("S1", {}).get("top") or [])}
        return ctx.trace


class Judge:
    """判断与决策分离。

    判断(grade)：这个用户有多危险 —— 只看 p 与敏感性区间，永远给得出来。
    决策(action)：现在该做什么 —— 阻断 / 预警 / 观察 / 放行，人工作为事后动作，
                  不再出现在在线状态机里。
    """

    @staticmethod
    def grade(ctx) -> str:
        p = ctx.get("FUSE", {}).get("p")
        if p is None:
            return "信息不足"
        if ctx.get("FAST", {}).get("hit") and p < .80:
            return "快通道告警"
        if p >= .80:
            return "高危" if ctx.get("FUSE", {}).get("p_lo", 0) >= .80 else "高危(敏感)"
        return "可疑" if p >= .20 else "低危"

    @staticmethod
    def decide(ctx) -> tuple[str, str]:
        q = ctx.q
        if not q.identity or q.q_call < .90:
            return "信息不足", "无"
        if not ctx.has("FUSE"):
            return "可疑", "观察"           # 预算耗尽：保守落地到观察，不阻断也不放行
        # 表6 第7行：任一分支因预算降级 → 不拿残缺向量下结论，保守落地到观察
        if any(ctx.get(k, {}).get("budget") for k in ("S1", "S2", "HAN")):
            return "可疑", "观察"
        f = ctx.get("FUSE", {})
        p, lo, hi = f.get("p", 0.0), f.get("p_lo", 0.0), f.get("p_hi", 1.0)
        capp = ctx.get("DQ", {}).get("level") == "degraded"   # 数据降质：禁止阻断
        if p >= .80:
            if lo >= .80 and not capp:
                return "高危", "阻断"
            return "高危(敏感)", "预警"
        if ctx.get("FAST", {}).get("hit"):
            return "快通道告警", "预警"
        # 表6 第5行：中风险、可观测历史不足 7 天、或敏感性上界越过高分阈值 → 可疑/观察
        if p >= .20 or ctx.sample.history_days < 7.0 or hi >= .80:
            return "可疑", "观察"
        return "低危", "放行"


def _np_step(sp: np.ndarray, te: np.ndarray, e: np.ndarray, W: dict) -> np.ndarray:
    """s_self/s_peer 已拼接的 256 维 + 16 + 12 -> 新记忆（纯 NumPy，供全局预扫描）。"""
    m1w, m1b = W["m1"]; m2w, m2b = W["m2"]
    d = np.concatenate([sp, te, e])
    m = np.maximum(d @ m1w + m1b, 0.0) @ m2w + m2b
    s_self = sp[:128]
    zw, zb = W["z"]; uzw, _ = W["uz"]
    z = 1 / (1 + np.exp(-(m @ zw + s_self @ uzw + zb)))
    rw, rb = W["r"]; urw, _ = W["ur"]
    r = 1 / (1 + np.exp(-(m @ rw + s_self @ urw + rb)))
    hw, hb = W["h"]; uhw, _ = W["uh"]
    h = np.tanh(m @ hw + (r * s_self) @ uhw + hb)
    return ((1 - z) * s_self + z * h).astype(np.float32)