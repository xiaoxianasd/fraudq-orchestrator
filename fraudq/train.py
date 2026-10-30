"""train — 分阶段训练并打包 artifacts/model.pt。

阶段：S2 模式库（无监督） → 轻筛（754 参数 BCE） → TGN 读出头（截断 BPTT）
      → HAN（BCE） → 15 维融合 + 三处 Platt 校准 → 测试集指标。
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .core import Budget, Context, Store, Trace
from .data import generate
from .s1 import S1Agent, TGN
from .s2 import S2Agent
from .s3 import Fusion, HAN, Platt, S3HanAgent
from .scheduler import Coordinator, Engine, Screener


def ctx_of(engine, q) -> Context:
    return Context(q, engine.store, engine.prepare(q), Budget(1e9), Trace(q.qid, q.user, q.cutoff), engine)


def fusion_inputs(engine, s1, s2, han_ag, fusion, q) -> Context:
    """按**推理路径**跑一遍分支，产出融合层的 18 维输入。

    必须与 Coordinator.run 完全一致，否则融合层学到的是另一个分布：
      * 调度器总先写 RELQ（数据质量 + 路径可用标记），训练若漏写，
        HAN 未执行时 v[6..14] 全为 0，而线上是 0.9x 量级；
      * 调度器只在 relation_trigger 成立时才跑 HAN，训练若见路径就跑，
        v[1]（HAN 对数几率）与 v[17]（HAN 执行掩码）会对不上（实测约 35% 的查询）。
    """
    from .s3 import FusionAgent
    c = ctx_of(engine, q)
    s1.run(c); s2.run(c)
    quals, cands, avail = han_ag.available(c)
    c.put("RELQ", quals=quals, avail=avail)              # 与调度器一致
    trig, _ = Coordinator.relation_trigger(c)
    if trig and any(avail.values()):                     # 与调度器一致
        han_ag.run(c)
    FusionAgent(fusion).run(c)
    return c


def split(queries, ratio=.7):
    """按用户 7:3 划分，并对「是否出现过正例」分层。

    纯随机划分在少数类用户只有十几个时会翻车：爆发型 10 个用户曾随机分成
    3 训练 / 7 留出，模型只见过 3 个爆发用户，指标会被划分方式左右。
    分层后正例用户与负例用户各自按比例切分，两边都留有各种形态。
    """
    users = sorted({q.user for q in queries})
    pos_u = {q.user for q in queries if q.label == 1}
    rng = np.random.default_rng(0)
    tr: set = set()
    for group in (sorted(pos_u), sorted(set(users) - pos_u)):
        rng.shuffle(group)
        tr |= set(group[:int(round(len(group) * ratio))])
    return [q for q in queries if q.user in tr], [q for q in queries if q.user not in tr]


def auc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean())


def fit(ds, epochs=5, out="artifacts/model.pt", verbose=True) -> dict:
    t0 = time.time()
    torch.manual_seed(0); np.random.seed(0)   # 固定随机初始化，保证结果可复现
    store = Store(ds.calls, ds.obs)
    # 统计量只在本数据域的用户上估计（外部号码不参与），且用户数不再写死为 200
    users = sorted(ds.archetype) if ds.archetype else sorted({c.caller for c in ds.calls})
    stats = store.fit_stats(users, max(q.cutoff for q in ds.queries))

    m: Dict[str, object] = {"tgn": TGN().float(), "han": HAN().float(), "fusion": Fusion().float(),
                            "screener": Screener().float(), "platt_T": Platt(), "platt_H": Platt(),
                            "platt_F": Platt(), "s2_norm": None, "s2_abn": None,
                            "s2_med": None, "s2_iqr": None, "version": "v1"}
    eng = Engine(store, m, peer_mode="global")
    tr, te = split(ds.queries)
    # 欠采样口径按说明书 3.5：正例全保留，**负例**上限 220。
    # 若按"总数 220"截断，正例一多（如 136 条）就只剩 84 条负例，训练集变成
    # 62% 正例，模型整体过判、精确率崩塌。
    rng = np.random.default_rng(1)
    pos = [q for q in tr if q.label == 1]
    neg = [q for q in tr if q.label == 0]
    if len(neg) > 220:
        neg = [neg[i] for i in rng.choice(len(neg), 220, replace=False)]
    tr_fit = pos + neg
    cutoffs = sorted({q.cutoff for q in ds.queries})
    if verbose:
        print(f"  [peers] {len(cutoffs)} 截点，按当前 TGN 权重惰性回放（不再预先生成随机权重快照）")

    # ---- S2 双模式库 ----
    s2 = S2Agent()
    feats, labs, bs_, bl_ = [], [], [], []
    for q in tr:
        c = ctx_of(eng, q); s2.run(c)
        d = c.get("S2")
        feats.append(d["pat"]); labs.append(q.label)
        bs_.append(d["B_s"]); bl_.append(d["B_l"])
    s2.fit(np.array(feats), np.array(labs), np.array(bs_), np.array(bl_))
    m.update({"s2_norm": torch.tensor(s2.norm, dtype=torch.float32),
              "s2_abn": torch.tensor(s2.abn, dtype=torch.float32),
              "s2_med": torch.tensor(s2.med, dtype=torch.float32),
              "s2_iqr": torch.tensor(s2.iqr, dtype=torch.float32),
              "s2_ok": bool(s2.bank_ok)})
    if verbose:
        print(f"  [S2] 正常库 {len(s2.norm)} 质心 | 异常库 {len(s2.abn)} 质心（爆发/渐进分层）")

    # ---- 轻筛 ----
    sc, X, y = m["screener"], None, torch.tensor([float(q.label) for q in tr])
    X = torch.tensor(np.stack([np.concatenate([eng.prepare(q).x, eng.prepare(q).last_event]) for q in tr]),
                     dtype=torch.float32)
    opt = torch.optim.Adam(sc.parameters(), lr=5e-3)
    for _ in range(600):
        mu, lv = sc(X)
        p_ = torch.sigmoid(mu)
        loss = nn.functional.binary_cross_entropy_with_logits(mu, y) +             .5 * (torch.exp(-lv.clamp(-6, 6)) * (p_ - y) ** 2 + lv.clamp(-6, 6)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if verbose:
        print(f"  [SCREEN] {sc.n_params()} 参数  loss={loss.item():.4f}")

    # ---- TGN 读出头 ----
    s1 = S1Agent(m["tgn"])
    opt = torch.optim.Adam(m["tgn"].parameters(), lr=2e-3)
    yy = torch.tensor([float(q.label) for q in tr_fit])
    for ep in range(epochs):
        eng._peer.clear()          # 每轮用当前权重重放对端记忆，避免训练/推理不一致
        eng._W = None              # 同时丢弃 NumPy 权重缓存，否则重放用的仍是第一轮的权重
        perm = np.random.default_rng(ep).permutation(len(tr_fit))
        tot, n = 0.0, 0
        for i in perm:
            c = ctx_of(eng, tr_fit[i])
            loss = nn.functional.binary_cross_entropy_with_logits(s1.logit(c), yy[i])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); n += 1
        if verbose:
            print(f"  [TGN] epoch {ep+1}/{epochs}  bce={tot/max(n,1):.4f}  {time.time()-t0:.0f}s")

    # ---- HAN ----
    han_ag = S3HanAgent(m["han"])
    opt = torch.optim.Adam(m["han"].parameters(), lr=3e-3)
    sub = [(q, i) for i, q in enumerate(tr_fit)]
    for ep in range(epochs):
        tot, n = 0.0, 0
        for q, i in sub:
            c = ctx_of(eng, q)
            quals, cands, avail = han_ag.available(c)
            if not any(avail.values()):
                continue
            logit = _han_logit(han_ag, c, quals, cands, avail)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, yy[i])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); n += 1
        if verbose and n:
            print(f"  [HAN] epoch {ep+1}/{epochs}  bce={tot/n:.4f} (可用 {n} 条)")

    # ---- 分支 Platt 校准 ----
    eng._peer.clear()              # 用收敛后的权重重放一次
    eng._W = None                  # 同上，否则快照仍来自训练早期转换的那份权重
    for key, pl in (("T", m["platt_T"]), ("H", m["platt_H"])):
        xs, ys2 = [], []
        for q, i in sub:
            c = ctx_of(eng, q)
            if key == "T":
                lg = s1.logit(c)
                xs.append(float(lg)); ys2.append(float(q.label))
            else:
                quals, cands, avail = han_ag.available(c)
                if any(avail.values()):
                    xs.append(float(_han_logit(han_ag, c, quals, cands, avail)))
                    ys2.append(float(q.label))
        if len(xs) > 10:
            X1 = torch.tensor(xs, dtype=torch.float32)
            Y1 = torch.tensor(ys2)
            opt2 = torch.optim.Adam(pl.parameters(), lr=5e-2)
            for _ in range(300):
                l2 = nn.functional.binary_cross_entropy_with_logits(
                    pl.lin(X1.unsqueeze(-1)).squeeze(-1), Y1)
                opt2.zero_grad(); l2.backward(); opt2.step()
            if verbose:
                print(f"  [PLATT-{key}] {len(xs)} 样本  最终 bce={l2.item():.4f}")

    # ---- 冻结上游，训练融合与校准 ----
    for p in m["tgn"].parameters(): p.requires_grad_(False)
    for p in m["han"].parameters(): p.requires_grad_(False)

    def branch_logits(q):
        return fusion_inputs(eng, s1, s2, han_ag, m["fusion"], q)

    rows, ys = [], []
    for q in tr_fit:
        c = branch_logits(q)
        f = c.get("FUSE")
        rows.append(f["v"])
        ys.append(float(q.label))
    V = torch.tensor(np.stack(rows), dtype=torch.float32)
    Y = torch.tensor(ys)
    opt = torch.optim.Adam(list(m["fusion"].parameters()) + list(m["platt_F"].parameters()), lr=1e-2)
    for _ in range(400):
        p = torch.sigmoid(m["platt_F"](m["fusion"](V)))
        loss = nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), Y)
        opt.zero_grad(); loss.backward(); opt.step()

    # Platt_T / Platt_H：一维逻辑回归校准
    for key, pl in (("T", m["platt_T"]), ("H", m["platt_H"])):
        pass

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"models": {k: (v.state_dict() if isinstance(v, nn.Module) else v) for k, v in m.items()},
                "peers": {c: {u: v for u, v in d.items()} for c, d in eng._peer.items()},
                "stats": stats, "banks": {"norm": m["s2_norm"], "abn": m["s2_abn"],
                                          "med": m["s2_med"], "iqr": m["s2_iqr"],
                                          "ok": m.get("s2_ok", True)},

                "version": "v1",
                "data_sig": {"users": len(ds.archetype), "calls": len(ds.calls),
                             "queries": len(ds.queries)}}, out)
    if verbose:
        print(f"  [SAVE] {out}  {os.path.getsize(out)/1e6:.1f} MB  {time.time()-t0:.0f}s")
    return {"engine": eng, "models": m, "test": te, "store": store}


def _han_logit(ag, ctx, quals, cands, avail):
    import torch as T
    feats, wts = [], []
    for k in ("D", "C", "I", "B", "P"):
        vs = list(cands[k])[:ag.max_nodes]
        if avail[k] and vs:
            feats.append(T.tensor(np.stack([ctx.engine.profile(v, ctx.q.cutoff) for v in vs]), dtype=T.float32))
            wts.append(T.tensor([cands[k][v] for v in vs], dtype=T.float32).clamp(min=1e-4))
        else:
            feats.append(T.zeros(0, 32)); wts.append(T.zeros(0))
    q = T.tensor([quals[k] for k in ("D", "C", "I", "B", "P")], dtype=T.float32)
    av = T.tensor([avail[k] for k in ("D", "C", "I", "B", "P")])
    x_u = T.tensor(ctx.sample.x, dtype=T.float32)
    lg, zH, beta = ag.han(feats, wts, q, av, x_u)
    ctx._han = (lg, zH, beta)
    return lg