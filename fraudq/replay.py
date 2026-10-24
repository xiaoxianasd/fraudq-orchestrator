"""replay — 原始证据组与检查点隔离重放（对端冻结、一阶局部）。"""
from __future__ import annotations

import numpy as np

from .core import REL_KINDS, REL_NAME


class ReplayAgent:
    name = "REPLAY"

    def estimate(self, ctx) -> float:
        return 20.0 + 0.5 * ctx.engine.cost_hint("REPLAY")

    def eligible(self, ctx) -> bool:
        f = ctx.get("FUSE")
        return bool(f) and (f["p"] >= .80 or self.gap(ctx) >= .30)

    @staticmethod
    def gap(ctx) -> float:
        s1, han = ctx.get("S1"), ctx.get("HAN")
        if not s1 or not han:
            return 0.0
        return abs(han.get("p_cal", 0) - s1.get("p_cal", 0))

    def groups(self, ctx) -> dict:
        u, t = ctx.q.user, ctx.q.cutoff
        vis = ctx.store.visible(u, t)
        g1 = {ctx.store.EID[u][k] for k in range(len(ctx.store.EID.get(u, [])))
              if ctx.store.E[u][k, 0] <= t and ctx.store.E[u][k, 1] >= t - 15 * 60_000
              and ctx.store.E[u][k, 2] == 1}
        out = {"g1": {"calls": g1, "keys": {}}}
        for i, k in enumerate(REL_KINDS, start=2):
            keys, _, _ = ctx.store.relations(u, t, k)
            out[f"g{i}"] = {"calls": set(), "keys": {k: set(keys)}}
        return out

    def run(self, ctx) -> None:
        p_full = float(ctx.get("FUSE").get("p", 0.0))
        g = self.groups(ctx)
        beta = ctx.get("HAN").get("beta") or [0] * 5
        order = list(REL_KINDS)
        has_g1 = bool(g["g1"]["calls"])
        rel = [f"g{i}" for i, k in enumerate(order, start=2) if g[f"g{i}"]["keys"].get(k)]
        # 权利要求6：第1组非空时优先保留，其余关系组按路径权重取最前若干个，
        # 再对前两组的并集重放一次，总场景数不超过 4 个。
        rel = sorted(rel, key=lambda x: -beta[int(x[1:]) - 2])[:2 if has_g1 else 3]
        scenes = {}
        if has_g1:
            scenes["g1"] = g["g1"]
        for r in rel:
            scenes[r] = g[r]
        if len(rel) >= 2:
            scenes["g_top2"] = {"calls": set(), "keys": {}}
            for k in REL_KINDS:
                scenes["g_top2"]["keys"][k] = (g[rel[0]]["keys"].get(k, set())
                                               | g[rel[1]]["keys"].get(k, set()))

        results = {}
        for name, sc in scenes.items():
            if not sc["calls"] and not any(sc["keys"].values()):
                continue
            p2 = ctx.engine.rescore(ctx, sc["calls"], sc["keys"])
            n_keys = sum(len(v) for v in sc["keys"].values())
            results[name] = {"p_without": round(p2, 3),
                             "delta": round(p_full - p2, 3), "n_keys": n_keys,
                             "calls": sorted(sc["calls"])[:20], "n": len(sc["calls"]) + n_keys}
        ctx.put("REPLAY", done=True, p_full=round(p_full, 3), scenes=results,
                best=max((r["delta"] for r in results.values()), default=0.0))
        ctx.trace.log("REPLAY", "p>=.80 or gap>=.30", f"{len(results)} 场景", ctx.elapsed(),
                      "Δ=" + " ".join(f"{k}:{v['delta']:+.3f}" for k, v in results.items()))