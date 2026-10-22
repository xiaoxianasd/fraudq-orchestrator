"""ingest — 批量数据接入：校验 → 幂等写入 → 缓存失效 → 自动重判。

对外只暴露两个函数和一个小封装类：

    ingest_rows(store, engine, rows, coord=None)      # 内存行（dict/DataFrame.to_dict）
    ingest_csv(store, engine, path, coord=None)       # CSV / JSONL 文件
    LiveSystem.load(model, data)                      # 一次性封装，供外部程序调用
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .core import BUSY, CONNECT, DAY, FAILED, UNANSWERED, Call, Obs, Query, Store

STATUS_MAP = {"接通": CONNECT, "connected": CONNECT, "0": CONNECT,
              "未接": UNANSWERED, "unanswered": UNANSWERED, "1": UNANSWERED,
              "忙线": BUSY, "busy": BUSY, "2": BUSY,
              "失败": FAILED, "failed": FAILED, "其他失败": FAILED, "3": FAILED}

# 允许的中英文字段别名
ALIAS = {"caller": ("caller", "主叫", "caller_no", "from"),
         "callee": ("callee", "被叫", "callee_no", "to"),
         "start": ("start", "start_at", "时间", "start_time"),
         "talk": ("talk", "talk_seconds", "时长", "duration"),
         "status": ("status", "状态"),
         "avail": ("available_at", "avail", "可用时间"),
         "cid": ("call_id", "cid", "编号"),
         "user": ("user", "用户", "phone_no_m"),
         "kind": ("kind", "类型"),
         "key": ("key", "标识", "device", "ip", "bs", "pay"),
         "obs_at": ("observed_at", "obs_at", "观测时间")}


def _pick(row: Dict[str, Any], field_name: str, default=None):
    for k in ALIAS[field_name]:
        if k in row and row[k] not in ("", None):
            return row[k]
    return default


@dataclass
class IngestReport:
    calls: int = 0
    obs: int = 0
    duplicated: int = 0
    invalid: List[dict] = field(default_factory=list)
    users: List[int] = field(default_factory=list)
    judgments: List[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {"calls": self.calls, "obs": self.obs, "duplicated": self.duplicated,
                "invalid": len(self.invalid), "users": len(self.users),
                "judgments": self.judgments}


def _norm_call(row: Dict[str, Any], seq: int):
    try:
        a = int(_pick(row, "caller")); b = int(_pick(row, "callee"))
    except (TypeError, ValueError):
        return None, "主叫/被叫缺失或非整数"
    if a <= 0 or b <= 0 or a == b:
        return None, "主叫/被叫非法（须为正整数且不相同）"
    talk = int(float(_pick(row, "talk", 0) or 0))
    if talk < 0:
        return None, "通话时长 < 0"
    st_raw = str(_pick(row, "status", CONNECT))
    if st_raw not in STATUS_MAP:
        return None, f"未知通话状态 {st_raw}"
    st = STATUS_MAP[st_raw]
    start = _pick(row, "start")
    if start is None:
        return None, "缺少通话开始时间"
    try:
        start = int(float(start))
    except (TypeError, ValueError):
        return None, "开始时间不是毫秒整数"
    avail = _pick(row, "avail")
    avail = int(float(avail)) if avail is not None else start + talk * 1000 + 500
    if avail < start:
        return None, "可用时间早于发生时间（违反可见性口径）"
    cid = str(_pick(row, "cid") or f"ing-{seq:08d}")
    return Call(cid, a, b, start, start + talk * 1000 + 500, avail, talk, st,
                talk > 0 or st != CONNECT, True, True), None


def _norm_obs(row: Dict[str, Any], seq: int):
    try:
        u = int(_pick(row, "user"))
    except (TypeError, ValueError):
        return None, "关联观测缺 user"
    kind = str(_pick(row, "kind", "D")).upper()
    if kind not in ("D", "I", "B", "P"):
        return None, f"未知观测类型 {kind}（仅 D/I/B/P）"
    key = _pick(row, "key")
    if not key:
        return None, "关联观测缺 key"
    at = _pick(row, "obs_at")
    if at is None:
        return None, "关联观测缺 observed_at"
    at = int(float(at))
    avail = _pick(row, "avail")
    avail = int(float(avail)) if avail is not None else at + 1000
    oid = str(_pick(row, "cid") or f"oing-{seq:08d}")
    return Obs(oid, u, kind, str(key), at, avail), None


def ingest_rows(store: Store, engine, rows: Iterable[Dict[str, Any]],
                coord=None, budget: float = 1000.0) -> IngestReport:
    """批量写入。返回结构化报告；coord 非空时对受影响用户自动重判。"""
    rep = IngestReport()
    affected, pre_existing = set(), set(store.CU)
    for i, raw in enumerate(rows):
        row = {str(k).strip(): v for k, v in raw.items()}
        if _pick(row, "caller") is not None:
            c, err = _norm_call(row, i)
            if err:
                rep.invalid.append({"row": i + 1, "error": err}); continue
            if store.add_call(c):
                rep.calls += 1
                affected.add(c.caller)                       # 主叫一定要重判
                if c.callee in pre_existing:                 # 被叫只在"本就是已知用户"时才重判
                    affected.add(c.callee)
            else:
                rep.duplicated += 1
        elif _pick(row, "user") is not None:
            o, err = _norm_obs(row, i)
            if err:
                rep.invalid.append({"row": i + 1, "error": err}); continue
            if store.add_obs(o):
                rep.obs += 1; affected.add(o.user)
            else:
                rep.duplicated += 1
        else:
            rep.invalid.append({"row": i + 1, "error": "无法识别：既没有 caller 也没有 user"})

    rep.users = sorted(affected)
    if affected:
        ts = max(store.latest(u) for u in affected)
        engine.invalidate(ts)
        if coord is not None:
            for u in rep.users:
                if store.history_days(u, store.latest(u)) < 0.05:
                    continue                                  # 全新用户没有历史，判了也无意义
                q = Query(f"ing-{u}-{ts}", u, store.latest(u), -1, .99, .95, True)
                tr = coord.run(q, budget, replay_mode="defer")
                f = tr.results.get("FUSE", {})
                rep.judgments.append({"user": u, "p": round(f.get("p", 0.0), 4),
                                      "grade": tr.status, "action": tr.action,
                                      "degraded": "FUSE" not in tr.results,
                                      "ms": round(tr.elapsed_ms, 1)})
    return rep


def ingest_csv(store: Store, engine, path: str, coord=None, budget: float = 1000.0) -> IngestReport:
    if path.lower().endswith(".jsonl"):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    else:
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    return ingest_rows(store, engine, rows, coord=coord, budget=budget)


class LiveSystem:
    """对外封装：加载 → 批量接入 → 查询。"""

    def __init__(self, ds, eng, coord):
        self.ds, self.eng, self.coord = ds, eng, coord

    @classmethod
    def load(cls, model: str, data: str):
        from .data import Dataset
        from .runtime import load as _load
        ds = Dataset.load(data)
        eng, coord = _load(model, ds)
        return cls(ds, eng, coord)

    def ingest(self, rows, rejudge: bool = True) -> IngestReport:
        return ingest_rows(self.eng.store, self.eng, rows,
                           coord=self.coord if rejudge else None)

    def ingest_file(self, path: str, rejudge: bool = True) -> IngestReport:
        return ingest_csv(self.eng.store, self.eng, path,
                          coord=self.coord if rejudge else None)

    def judge(self, user: int, cutoff: Optional[int] = None) -> dict:
        t = cutoff or self.eng.store.latest(user)
        tr = self.coord.run(Query(f"live-{user}-{t}", user, t, -1, .99, .95, True),
                            200.0, replay_mode="defer")
        return {"user": user, "p": tr.results.get("FUSE", {}).get("p", 0.0),
                "grade": tr.status, "action": tr.action, "ms": tr.elapsed_ms}