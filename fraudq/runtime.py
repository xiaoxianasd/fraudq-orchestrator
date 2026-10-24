"""runtime — 加载工件、批量评估、单次查询。"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch

from .core import Store
from .data import Dataset
from .s1 import TGN
from .s3 import Fusion, HAN, Platt
from .scheduler import Coordinator, Engine, Screener


def load(path: str, ds: Dataset) -> tuple[Engine, Coordinator]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = {"tgn": TGN(), "han": HAN(), "fusion": Fusion(), "screener": Screener(),
         "platt_T": Platt(), "platt_H": Platt(), "platt_F": Platt()}
    for k, mod in list(m.items()):
        mod.load_state_dict(blob["models"][k])
        mod.eval()
    b = blob["banks"]
    m.update({"s2_norm": b["norm"].numpy(), "s2_abn": b["abn"].numpy(),
              "s2_med": b["med"].numpy(), "s2_iqr": b["iqr"].numpy(),
              "s2_ok": bool(b.get("ok", True)), "version": blob.get("version", "v1")})
    st = blob["stats"]
    store = Store(ds.calls, ds.obs, {"med": np.array(st["med"]), "iqr": np.array(st["iqr"])})
    eng = Engine(store, m)
    # 工件里的对端记忆与最终权一致（训练每轮重放、保存前再重放一次），可直接复用；
    # 新截点则按当前权重现场回放，二者结果相同。
    if blob.get("peers"):
        eng._peer = {int(k): v for k, v in blob["peers"].items()}
    return eng, Coordinator(eng)


def auc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean())


def batch(coord: Coordinator, queries, budget: float = 200.0, replay_mode: str = "defer",
          finalize: bool = False) -> dict:
    """replay_mode='defer' 时只统计在线时延；finalize=True 再补跑异步复核。"""
    rows, t0 = [], time.perf_counter()
    for q in queries:
        tr = coord.run(q, budget, replay_mode=replay_mode)
        f = tr.results.get("FUSE", {})
        p = f.get("p", tr.results.get("SCREEN", {}).get("p0", 0.0))
        online = tr.elapsed_ms
        rms, final_status = 0.0, tr.status
        if finalize and tr.results.get("REPLAY", {}).get("pending"):
            tr, rms = coord.finalize(tr)
            final_status = tr.status
        rows.append({"q": q, "p": p, "status": final_status, "online_status": tr.status,
                     "ms": online, "replay_ms": rms, "trace": tr})
    y = [r["q"].label for r in rows]
    p = [r["p"] for r in rows]
    ms = [r["ms"] for r in rows]
    thr = .80                       # 与判决阈值一致（表6 的高分阈值）
    pred = [1 if x >= thr else 0 for x in p]
    tp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    from collections import Counter
    rep_ms = [r["replay_ms"] for r in rows if r["replay_ms"] > 0]
    return {"rows": rows, "auc": auc(y, p), "precision": prec, "recall": rec,
            "deferred": sum(1 for r in rows if r["replay_ms"] > 0),
            "replay_p50": float(np.percentile(rep_ms, 50)) if rep_ms else 0.0,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
            "total_s": time.perf_counter() - t0,
            "status": Counter(r["status"] for r in rows)}