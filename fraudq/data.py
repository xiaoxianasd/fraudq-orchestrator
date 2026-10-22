"""data — 合成 CDR 数据集（爆发型 / 慢热型 / 团伙型 + 正常）与 CSV 适配。

公开数据集调研结论（2026-10）：CCL2023-FCC、天池"电信诈骗五分类"均为**案情文本
分类**，Amazon fraud-dataset-benchmark 为电商交易，均不含"通话记录 + 五类关联
观测 + 截点标签"的三元结构；真实 CDR 受法规限制不公开——这正是本发明的前提。
故本模块按专利字段口径生成带真值的合成集，并保留 CSV 适配器以接入任何同构真实数据。
"""
from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .core import BUSY, CONNECT, DAY, FAILED, HOUR, MIN, Obs, Call, Query, UNANSWERED

VICTIM_BASE = 900000
OUT_POOL = [200000 + i for i in range(1200)]   # 有限外部号码池，避免节点数爆炸


@dataclass
class Dataset:
    calls: List[Call]
    obs: List[Obs]
    queries: List[Query]
    archetype: Dict[int, str]
    onset: Dict[int, int]
    origin: int = 0

    def stats(self) -> dict:
        users = sorted({c.caller for c in self.calls} | {c.callee for c in self.calls})
        pos = sum(1 for q in self.queries if q.label == 1)
        return {
            "users": len(users), "calls": len(self.calls), "obs": len(self.obs),
            "queries": len(self.queries), "positive": pos,
            "negatives": len(self.queries) - pos,
            "fraud_users": sum(1 for v in self.archetype.values() if v != "normal"),
            "kinds": {k: sum(1 for o in self.obs if o.kind == k) for k in ("D", "I", "B", "P")},
        }

    # ---------- CSV ----------
    def save(self, out: str) -> None:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "calls.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["call_id", "caller", "callee", "start_at", "end_at", "available_at",
                        "talk_seconds", "status", "duration_valid", "caller_ok", "callee_ok", "batch"])
            for c in self.calls:
                w.writerow([c.cid, c.caller, c.callee, c.start, c.end, c.avail, c.talk, c.status,
                            int(c.dur_valid), int(c.caller_ok), int(c.callee_ok), c.batch])
        with open(os.path.join(out, "obs.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["obs_id", "user", "kind", "key", "observed_at", "available_at", "valid"])
            for o in self.obs:
                w.writerow([o.oid, o.user, o.kind, o.key, o.at, o.avail, int(o.valid)])
        with open(os.path.join(out, "queries.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "user", "cutoff", "label", "q_call", "q_d", "identity_known"])
            for q in self.queries:
                w.writerow([q.qid, q.user, q.cutoff, q.label, f"{q.q_call:.4f}",
                            f"{q.q_d:.4f}", int(q.identity)])
        with open(os.path.join(out, "truth.json"), "w", encoding="utf-8") as f:
            json.dump({"archetype": {str(k): v for k, v in self.archetype.items()},
                       "onset": {str(k): v for k, v in self.onset.items()}}, f, ensure_ascii=False)

    @classmethod
    def load(cls, src: str) -> "Dataset":
        def rows(name):
            with open(os.path.join(src, name), encoding="utf-8") as f:
                return list(csv.DictReader(f))
        calls = [Call(r["call_id"], int(r["caller"]), int(r["callee"]), int(r["start_at"]),
                      int(r["end_at"]), int(r["available_at"]), int(r["talk_seconds"]),
                      int(r["status"]), r["duration_valid"] in ("1", "True"),
                      r["caller_ok"] in ("1", "True"), r["callee_ok"] in ("1", "True"),
                      int(r.get("batch", 0) or 0)) for r in rows("calls.csv")]
        obs = [Obs(r["obs_id"], int(r["user"]), r["kind"], r["key"], int(r["observed_at"]),
                   int(r["available_at"]), r["valid"] in ("1", "True")) for r in rows("obs.csv")]
        qs = [Query(r["query_id"], int(r["user"]), int(r["cutoff"]), int(r["label"]),
                    float(r["q_call"]), float(r["q_d"]), r["identity_known"] in ("1", "True"))
              for r in rows("queries.csv")]
        p = os.path.join(src, "truth.json")
        truth = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"archetype": {}, "onset": {}}
        return cls(calls, obs, qs, {int(k): v for k, v in truth["archetype"].items()},
                   {int(k): v for k, v in truth["onset"].items()})


def _diurnal(rng: random.Random, day: int) -> int:
    """返回一天内的呼叫时刻（毫秒），双峰分布。"""
    r = rng.random()
    if r < .35:
        h = rng.gauss(10.5, 2.2)
    elif r < .85:
        h = rng.gauss(19.5, 2.6)
    else:
        h = rng.gauss(1.5, 1.2)
    h = min(23.99, max(0.0, h))
    return day * DAY + int(h * HOUR)


def generate(seed: int = 7, days: int = 30, n_normal: int = 180,
             n_burst: int = 10, n_slow: int = 10, n_ring: int = 2, ring_size: int = 3) -> Dataset:
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)
    uid = 0
    users: Dict[int, str] = {}
    lam: Dict[int, float] = {}
    onset: Dict[int, int] = {}
    T = days * DAY
    TEND = (days - 1) * DAY + 21 * HOUR     # 最后一个查询截点：当地 21:00

    def new_user(kind: str, rate: float, on: int = T) -> int:
        nonlocal uid
        users[uid] = kind
        lam[uid] = rate
        onset[uid] = on
        uid += 1
        return uid - 1

    for _ in range(n_normal):                       # 活跃度跨三个数量级
        new_user("normal", float(npr.lognormal(1.9, .95)) / 24.0)
    rings = []
    for _ in range(n_ring):
        ring = [new_user("ring", .55, TEND - 10 * DAY) for _ in range(ring_size)]
        rings.append(ring)
    for _ in range(n_burst):
        new_user("burst", .12, TEND - int(rng.uniform(.55, .9) * HOUR))
    for _ in range(n_slow):
        new_user("slow", .18, TEND - int(rng.uniform(10, 13) * DAY))

    calls: List[Call] = []
    cid = 0

    def add(a: int, b: int, t: int, talk: int, st: int, dv: bool, av_off: int = 0) -> None:
        nonlocal cid
        calls.append(Call(f"c{cid:07d}", a, b, t, t + int(talk) * 1000 + 500,
                          t + int(talk) * 1000 + 500 + av_off, int(talk), st, dv, True, True, batch=0))
        cid += 1

    friend_pool = {}

    for u, kind in users.items():
        n = int(npr.poisson(lam[u] * T / HOUR))
        n = min(n, 1400)
        friends = [int(x) for x in npr.choice([k for k in users if k != u], size=min(8, len(users) - 1), replace=False)]
        friend_pool[u] = friends
        for _ in range(n):
            t = _diurnal(rng, rng.randrange(days))
            if rng.random() < .55:
                b = rng.choice(friends)
            else:
                b = rng.choice(OUT_POOL)
            p_con = .72 if rng.random() < .7 else .35
            if rng.random() < p_con:
                add(u, b, t, int(rng.gauss(48, 35)), CONNECT, True, rng.randint(200, 600_000))
            else:
                st = rng.choices([UNANSWERED, BUSY, FAILED], [.7, .2, .1])[0]
                add(u, b, t, 0, st, rng.random() < .8, rng.randint(200, 600_000))

    # ---- 爆发型：短时窗口内 60~120 通短通话，目标全为新号码 ----
    for u, kind in users.items():
        if kind != "burst":
            continue
        on = onset[u]
        k = int(rng.randint(60, 120))
        for i in range(k):
            t = on + int(i * rng.uniform(0.4, 0.9) * MIN)
            if t >= TEND:
                break
            b = VICTIM_BASE + int(rng.uniform(0, 400))
            add(u, b, t, int(rng.uniform(3, 22)), CONNECT if rng.random() < .8 else UNANSWERED,
                True, rng.randint(100, 3000))

    # ---- 慢热型：12 天内日均 5→45 通，目标逐日扩张，时段逐渐后移 ----
    for u, kind in users.items():
        if kind != "slow":
            continue
        on = onset[u]
        for d in range(0, 13):
            base = on + d * DAY
            if base >= T:
                break
            k = 5 + d * 3
            for i in range(k):
                t = base + int((19.0 + min(3.5, d * .25) + rng.gauss(0, 1.2)) * HOUR)
                if t >= T:
                    continue
                b = VICTIM_BASE + int(rng.uniform(0, 400))
                add(u, b, t, int(rng.uniform(8, 60)), CONNECT if rng.random() < .6 else UNANSWERED,
                    True, rng.randint(300, 20_000))

    # ---- 团伙型：成员互打 + 共呼同一批受害者 ----
    for ring in rings:
        for u in ring:
            for d in range(days):
                for i in range(int(rng.uniform(2, 8))):
                    t = d * DAY + int(rng.gauss(20, 2) * HOUR)
                    if not (0 <= t < T):
                        continue
                    b = rng.choice([x for x in ring if x != u]) if rng.random() < .25 \
                        else VICTIM_BASE + int(rng.uniform(0, 60))
                    add(u, b, t, int(rng.uniform(20, 90)), CONNECT if rng.random() < .8 else BUSY,
                        True, rng.randint(200, 5000))

    # ---- 关系观测 ----
    obs: List[Obs] = []
    oid = 0

    def add_obs(u: int, kind: str, key: str, t: int, delay: int = 5000) -> None:
        nonlocal oid
        obs.append(Obs(f"o{oid:07d}", u, kind, key, t, t + delay))
        oid += 1

    def keys_for(u: int) -> Dict[str, str]:
        k = users[u]
        if k == "ring":
            r = next(r for r in rings if u in r)
            return {"D": f"dev-ring{r[0]}", "I": f"ip-ring{r[0]}", "B": f"bs-ring{r[0]}", "P": f"pay-ring{r[0]}"}
        if k == "burst":
            return {"D": f"dev-u{u}", "I": f"ip-u{u}", "B": f"bs-u{u}", "P": f"pay-u{u}"}
        venue = u % 40 if k == "normal" else u
        return {"D": f"dev-u{u}", "I": f"ip-v{venue}", "B": f"bs-v{venue}", "P": f"pay-u{u}"}

    for c in calls:                              # 通话级基站观测，使 B 路径 1 小时窗有样本
        if rng.random() < .40:
            base = keys_for(c.caller)["B"]
            add_obs(c.caller, "B", base if rng.random() < .85 else base + "-x", c.start,
                    rng.randint(500, 3000))

    for u, kind in users.items():
        base = keys_for(u)
        for kd in ("D", "I", "B", "P"):
            step = 2 * DAY if kd in ("D", "P") else 3 * HOUR
            t = rng.randrange(step)
            while t < T:
                hot = kind in ("normal", "ring") or t < onset.get(u, T)
                key = base[kd] if hot else f"{base[kd]}-n{t // DAY}"
                add_obs(u, kd, key, t, rng.randint(1000, 60_000))
                t += step + rng.randint(0, step // 2)
        if kind == "burst":                    # 爆发前 90 分钟密集换号换机
            for i in range(3):
                t = onset[u] + i * 8 * MIN
                add_obs(u, "D", f"dev-u{u}-hot{i}", t, 2000)
                add_obs(u, "B", f"bs-u{u}-hop{i}", t, 2000)
                add_obs(u, "I", f"ip-u{u}-hot{i}", t, 2000)
            add_obs(u, "P", f"pay-u{u}-new", onset[u] + 5 * MIN, 2000)

    # ---- 查询截点 ----
    queries: List[Query] = []
    qn = 0
    cuts = [TEND - 7 * DAY, TEND - 2 * DAY, TEND]
    for u, kind in users.items():
        for c in cuts:
            if c <= DAY:
                continue
            label = int(kind != "normal" and c >= onset.get(u, T + 1))
            qc = 0.990 + rng.uniform(0, .009)
            if rng.random() < .04:
                qc = 0.94
            queries.append(Query(f"q{qn:05d}", u, c, label, round(min(.999, max(.90, qc)), 4),
                                 round(0.80 + rng.uniform(0, .18), 4), rng.random() > .01))
            qn += 1
    return Dataset(calls, obs, queries, users, onset)
