"""s1 — 行为语义智能体：32 维画像 + 128 维 TGN 记忆 + 4 头注意力读出。"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from .core import DAY, HOUR, SEC, Call

TAU = torch.exp(torch.arange(8, dtype=torch.float32) / 7 * math.log(30 * 86400))


def time_enc(dt: torch.Tensor) -> torch.Tensor:
    """dt: (B,) 秒 -> (B,16) 正余弦时间编码。"""
    x = dt.clamp(min=0).unsqueeze(-1) / TAU
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


def te1(v: float) -> torch.Tensor:
    """标量时间差 -> (16,) 编码。"""
    return time_enc(torch.tensor([float(v)]))[0]


class TGN(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.msg = nn.Sequential(nn.Linear(2 * dim + 16 + 12, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.wz, self.uz = nn.Linear(dim, dim), nn.Linear(dim, dim, bias=False)
        self.wr, self.ur = nn.Linear(dim, dim), nn.Linear(dim, dim, bias=False)
        self.wh, self.uh = nn.Linear(dim, dim), nn.Linear(dim, dim, bias=False)
        self.base = nn.Linear(dim + 32, 64)
        self.q = nn.ModuleList(nn.Linear(80, 16, bias=False) for _ in range(4))
        self.k = nn.ModuleList(nn.Linear(92, 16, bias=False) for _ in range(4))
        self.v = nn.ModuleList(nn.Linear(92, 16, bias=False) for _ in range(4))
        self.out = nn.Linear(128, 64)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    # 单事件记忆更新：d = [s_self; ρ·s_peer; TE(Δt); e] ∈ R^284
    def step(self, s_self: torch.Tensor, s_peer: torch.Tensor, dt: torch.Tensor,
             e: torch.Tensor) -> torch.Tensor:
        d = torch.cat([s_self, s_peer, time_enc(dt), e], dim=-1)
        m = self.msg(d)
        z = torch.sigmoid(self.wz(m) + self.uz(s_self))
        r = torch.sigmoid(self.wr(m) + self.ur(s_self))
        h = torch.tanh(self.wh(m) + self.uh(r * s_self))
        return (1 - z) * s_self + z * h

    def score(self, s_u, x_u, h_nb, e_nb, dt_nb, valid_nb) -> torch.Tensor:
        hu = torch.relu(self.base(torch.cat([s_u, x_u], -1)))            # (64,)
        ctx = torch.cat([h_nb, e_nb, dt_nb], -1)                          # (N,92)
        outs = []
        for i in range(4):
            qi = self.q[i](torch.cat([hu, te1(0.0)], -1))
            ki, vi = self.k[i](ctx), self.v[i](ctx)
            a = (ki @ qi) / math.sqrt(16)
            a = a.masked_fill(~valid_nb, -1e9).softmax(0)
            outs.append(a @ vi)
        nb = torch.cat(outs, -1)                                          # (64,)
        z = torch.relu(self.out(torch.cat([hu, nb], -1)))                 # (64,)
        return self.head(z).squeeze(-1)


class S1Agent:
    """行为语义智能体。输出 p_T / p^_T / C_sem / 贡献特征 / 记忆检查点。"""

    name = "S1"

    def __init__(self, model: TGN, ctx_len: int = 20, window: int = 64):
        self.model = model
        self.ctx_len = ctx_len
        self.window = window

    def estimate(self, ctx) -> float:
        return 1.2 * max(ctx.engine.cost_hint("S1"), 6.0)

    def eligible(self, ctx) -> bool:
        return ctx.sample.n_ev > 0

    def skip(self, ctx, reason: str) -> None:
        ctx.put("S1", skipped=True, reason=reason, p=0.5, p_cal=0.5, c_sem=0.0,
                n_ev=0, top=[], checkpoints=0, cps=[])

    def stream(self, ctx) -> tuple[List[torch.Tensor], List[int]]:
        """目标用户本地流（对端记忆冻结）：逐事件记忆快照 + 检查点下标。"""
        u, t = ctx.q.user, ctx.q.cutoff
        E, EC = ctx.store.E[u], ctx.store.EC[u]
        idxs = np.flatnonzero(ctx.store.vis_mask(u, t, ctx.skip_calls))[-self.window:]
        peers = ctx.engine.peer_memory(t)
        s = torch.zeros(128)
        snaps, cps = [], []
        last = None
        for k in idxs:
            av, st, role, status, talk, peer = E[k]
            ev = torch.tensor(ctx.store.event_vector(EC[k], u, t), dtype=torch.float32)
            dt = torch.tensor(0.0 if last is None else max(0.0, (int(av) - last) / 1000.0))
            s_peer = peers.get(int(peer), torch.zeros(128))
            gate = 1.0 if role == 1 else float(status == 0 and talk * 1000 >= 5000)
            s = self.model.step(s, gate * s_peer, dt, ev)
            snaps.append(s)
            cps.append(int(k))
            last = int(av)
            if len(cps) % 64 == 0:
                s = s.detach()
        return snaps, cps

    def logit(self, ctx):
        """返回可微分的标量 logit（训练与推理共用同一条前向路径）。"""
        m = self.model
        u, t = ctx.q.user, ctx.q.cutoff
        snaps, cps = self.stream(ctx)
        s_u = snaps[-1] if snaps else torch.zeros(128)
        x = torch.tensor(ctx.sample.x, dtype=torch.float32)
        peers = ctx.engine.peer_memory(t)
        rows, evs, dts = [], [], []
        for k in cps[-self.ctx_len:]:
            av, st, role, status, talk, peer = ctx.store.E[u][k]
            rows.append(ctx.engine.peer_h(int(peer), t))
            evs.append(torch.tensor(ctx.store.event_vector(ctx.store.EC[u][k], u, t), dtype=torch.float32))
            dts.append(te1(max(0.0, (t - int(av)) / 1000.0)))
        if rows:
            h_nb, e_nb, dt_nb = torch.stack(rows), torch.stack(evs), torch.stack(dts)
            valid = torch.ones(len(rows), dtype=torch.bool)
        else:
            h_nb, e_nb, dt_nb = torch.zeros(0, 64), torch.zeros(0, 12), torch.zeros(0, 16)
            valid = torch.zeros(0, dtype=torch.bool)
        ctx._n_snap = len(snaps)
        ctx._cps = cps
        return m.score(s_u, x, h_nb, e_nb, dt_nb, valid)

    def run(self, ctx) -> None:
        logit = self.logit(ctx)
        p = float(torch.sigmoid(logit))
        p_cal = float(ctx.engine.platt("T", logit.unsqueeze(0)))
        n = getattr(ctx, "_n_snap", 0)
        c_sem = 0.5 * float(ctx.sample.m.sum() / 16.0) + 0.5 * min(1.0, n / 20.0)
        cps = getattr(ctx, "_cps", [])
        ctx.put("S1", p=p, p_cal=p_cal, c_sem=c_sem, n_ev=n, top=self._top(ctx), checkpoints=n,
                cps=list(cps[-20:]), logit=float(logit))
        ctx.trace.log("S1", "n_ev>0", "run", ctx.elapsed(),
                      f"p_T={p:.3f} p^_T={p_cal:.3f} C_sem={c_sem:.2f} 检查点={n}")

    def _top(self, ctx) -> list:
        z = np.abs(ctx.sample.x[:16])
        return [(int(i), round(float(z[i]), 2)) for i in np.argsort(-z)[:3]]