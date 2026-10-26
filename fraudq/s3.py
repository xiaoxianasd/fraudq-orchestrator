"""s3 — 研判处置智能体：五元路径 HAN + 15 维可信度自适应融合。"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .core import REL_GATE, REL_KINDS, REL_NAME, REL_WINDOW


class Platt(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        with torch.no_grad():
            self.lin.weight.fill_(1.0)
            self.lin.bias.zero_()

    def forward(self, x):
        return torch.sigmoid(self.lin(x.unsqueeze(-1)).squeeze(-1))


class HAN(nn.Module):
    """五条元路径不共享参数；可用掩码 + 路径质量门控的语义注意力。"""

    def __init__(self, heads: int = 4, hdim: int = 16):
        super().__init__()
        self.heads, self.hdim = heads, hdim
        self.proj = nn.ModuleList([nn.ModuleList([nn.Linear(32, hdim) for _ in range(heads)])
                                   for _ in REL_KINDS])
        self.att = nn.Parameter(torch.zeros(len(REL_KINDS), heads, hdim))
        self.sem = nn.Linear(64, 32)
        self.qs = nn.Parameter(torch.zeros(len(REL_KINDS), 32))
        self.head = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, feats: List[torch.Tensor], wts: List[torch.Tensor],
                quals: torch.Tensor, avail: torch.Tensor, x_u: torch.Tensor):
        zs, sem = [], []
        for p in range(len(REL_KINDS)):
            if not avail[p] or feats[p].numel() == 0:
                zs.append(torch.zeros(64))
                sem.append(torch.tensor(-1e9))
                continue
            outs = []
            for h in range(self.heads):
                t = torch.tanh(self.proj[p][h](feats[p]))              # (N,16)
                s = t @ self.att[p, h]                                  # (N,)
                s = s + torch.log(wts[p] + 1e-6)
                alpha = s.softmax(0)
                outs.append(alpha @ t)
            z = torch.cat(outs)                                         # (64,)
            zs.append(z)
            sem.append(self.qs[p] @ torch.tanh(self.sem(z)) + torch.log(quals[p] + 1e-6))
        zs_t, sem_t = torch.stack(zs), torch.stack(sem)
        beta = sem_t.softmax(0)
        zH = (beta.unsqueeze(-1) * zs_t).sum(0)
        return self.head(torch.cat([zH, x_u])).squeeze(-1), zH, beta


class Fusion(nn.Module):
    """15 维可信度自适应融合层。"""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(18, 1)

    def forward(self, v):
        return self.lin(v).squeeze(-1)


class S3HanAgent:
    name = "HAN"

    def __init__(self, han: HAN, max_nodes: int = 31):
        self.han = han
        self.max_nodes = max_nodes

    def estimate(self, ctx) -> float:
        return 1.2 * max(ctx.engine.cost_hint("HAN"), 5.0)

    def eligible(self, ctx) -> bool:
        return True

    def quality(self, ctx) -> Dict[str, float]:
        """四类关系质量 = 窗口内有效且可匹配记录数 / 该用户窗口内可用通话数。

        与说明书 3.2、表4 一致；分母不足 5 条时质量记为未知（0），该路径不可用。
        共同目标路径质量直接取话单完整性。
        """
        st, q = ctx.store, ctx.q
        calls = st.visible(q.user, q.cutoff)
        out = {"C": float(q.q_call)}
        for k in ("D", "I", "B", "P"):
            win = REL_WINDOW[k]
            n_call = int((calls[:, 1] >= q.cutoff - win).sum()) if len(calls) else 0
            n_obs = sum(1 for _key, at, av in st.RU[k].get(q.user, ())
                        if av <= q.cutoff and at >= q.cutoff - win)
            out[k] = min(1.0, n_obs / n_call) if n_call >= 5 else 0.0
        return out

    def candidates(self, ctx) -> Dict[str, Dict[int, float]]:
        out = {}
        for k in REL_KINDS:
            w = ctx.store.relation_weight(ctx.q.user, ctx.q.cutoff, k,
                                          frozenset(ctx.skip_keys.get(k, ())))
            if len(w) > self.max_nodes:
                top = sorted(w.items(), key=lambda kv: -kv[1])[:self.max_nodes]
                w = dict(top)
            out[k] = w
        return out

    def available(self, ctx) -> tuple[dict, dict, dict]:
        quals = self.quality(ctx)
        cands = self.candidates(ctx)
        avail = {k: (quals[k] >= REL_GATE[k] and len(cands[k]) > 0) for k in REL_KINDS}
        return quals, cands, avail

    def run(self, ctx) -> None:
        quals, cands, avail = self.available(ctx)
        feats, wts = [], []
        for k in REL_KINDS:
            vs = list(cands[k])[:self.max_nodes]
            if avail[k] and vs:
                f = torch.tensor(np.stack([ctx.engine.profile(v, ctx.q.cutoff) for v in vs]),
                                 dtype=torch.float32)
                w = torch.tensor([cands[k][v] for v in vs], dtype=torch.float32)
                feats.append(f)
                wts.append(w.clamp(min=1e-4))
            else:
                feats.append(torch.zeros(0, 32))
                wts.append(torch.zeros(0))
        q = torch.tensor([quals[k] for k in REL_KINDS], dtype=torch.float32)
        av = torch.tensor([avail[k] for k in REL_KINDS])
        x_u = torch.tensor(ctx.sample.x, dtype=torch.float32)
        logit, zH, beta = self.han(feats, wts, q, av, x_u)
        p = float(torch.sigmoid(logit))
        p_cal = ctx.engine.platt("H", logit.unsqueeze(0)).item()
        ctx.put("HAN", p=p, p_cal=p_cal, beta=beta.detach().tolist(),
                avail={k: bool(avail[k]) for k in REL_KINDS},
                quals={k: round(quals[k], 3) for k in REL_KINDS},
                n_nb={k: len(cands[k]) for k in REL_KINDS}, z=zH.detach())
        ctx.trace.log("HAN", "relation_trigger & any(m_φ)", "run", ctx.elapsed(),
                      f"p^_H={p_cal:.3f} 可用路径={[k for k in REL_KINDS if avail[k]]}")


SENS_LO, SENS_HI = 0.02, 0.98     # 敏感性区间把关系分支推到的两个极端值


class FusionAgent:
    name = "FUSE"

    def __init__(self, fusion: Fusion):
        self.fusion = fusion

    def estimate(self, ctx) -> float:
        return 0.5

    def eligible(self, ctx) -> bool:
        return True

    def run(self, ctx) -> None:
        s1 = ctx.get("S1")
        s2 = ctx.get("S2")
        han = ctx.get("HAN")
        # 优先读调度器落的 RELQ（与 HAN 是否执行无关），无则退回 HAN 结果
        rel = ctx.get("RELQ") or han
        quals = rel.get("quals") or {k: 0.0 for k in REL_KINDS}
        av = rel.get("avail") or {k: False for k in REL_KINDS}
        v = np.zeros(18)
        v[0] = self._logit(s1.get("p_cal", 0.5))
        v[1] = self._logit(han.get("p_cal", 0.5)) if han else 0.0
        v[2] = s2.get("E_ts", 0.0) if s2 else 0.0
        v[3] = s1.get("c_sem", 0.0)
        v[4] = s2.get("C_temp", 0.0) if s2 else 0.0
        v[5] = ctx.q.q_call
        v[6] = quals.get("D", 0.0)
        v[7] = quals.get("I", 0.0)
        v[8] = quals.get("B", 0.0)
        v[9] = quals.get("P", 0.0)
        for i, k in enumerate(REL_KINDS):
            v[10 + i] = 1.0 if av.get(k) else 0.0
        v[15] = 1.0 if s1 else 0.0      # 行为语义分支执行掩码
        v[16] = 1.0 if s2 else 0.0      # 时序模式核验分支执行掩码
        v[17] = 1.0 if han else 0.0     # 异构关系核查执行掩码
        logit = self.fusion(torch.tensor(v, dtype=torch.float32))
        # 判决分数统一用校准后的 p（训练目标就是 platt_F(σ(z))，两者必须同一把尺子）
        p_cal = float(ctx.engine.platt("F", logit.unsqueeze(0)))
        # 敏感性区间：把关系分支推到两个极端，看综合分会落到哪（<1ms，无需重跑分支）
        lo = v.copy(); lo[1] = self._logit(SENS_LO)
        hi = v.copy(); hi[1] = self._logit(SENS_HI)
        with torch.no_grad():
            f = self.fusion
            lo_l = f(torch.tensor(lo, dtype=torch.float32))
            hi_l = f(torch.tensor(hi, dtype=torch.float32))
            p_lo = float(ctx.engine.platt("F", lo_l))
            p_hi = float(ctx.engine.platt("F", hi_l))
        ctx.put("FUSE", p=p_cal, p_raw=float(torch.sigmoid(logit)), v=v, logit=float(logit),
                p_lo=min(p_lo, p_hi), p_hi=max(p_lo, p_hi), p_cal=p_cal)
        ctx.trace.log("FUSE", "always", "run", ctx.elapsed(), f"p={p_cal:.3f}")

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-3), 1 - 1e-3)
        return math.log(p / (1 - p))