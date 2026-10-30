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
CUST_BASE = 700000                             # 合法高频业务（客服/催收/坐席）的存量客户池
LEAD_BASE = 600000                             # 外呼线索池：只打一次的新名单，占比高但不重复
COVER_P = .88          # 每条通话为两端各生成四类关联观测的概率（保证窗口覆盖率 >0.80）
OUT_POOL = [200000 + i for i in range(1200)]   # 有限外部号码池，避免节点数爆炸
FRAUD_KINDS = ("burst", "slow", "ring", "surg")   # 正例；service/family/growth/event 是难负例


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
            "fraud_users": sum(1 for v in self.archetype.values() if v in FRAUD_KINDS),
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
            w.writerow(["obs_id", "user", "kind", "key", "observed_at", "available_at", "valid", "etype"])
            for o in self.obs:
                w.writerow([o.oid, o.user, o.kind, o.key, o.at, o.avail, int(o.valid), o.etype])
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
                   int(r["available_at"]), r["valid"] in ("1", "True"),
                   (r.get("etype") or "account")) for r in rows("obs.csv")]
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


def generate(seed: int = 7, days: int = 30, n_normal: int = 240,
             n_burst: int = 16, n_slow: int = 16, n_ring: int = 3, ring_size: int = 3,
             n_service: int = 10, n_family: int = 8, family_size: int = 4,
             n_growth: int = 10, n_event: int = 12, n_surg: int = 10) -> Dataset:
    """合成 CDR。

    前 6 个参数与旧版一致：把 n_service / n_family / n_growth 置 0 即可逐字节复现旧数据集。
    后三类是**难负例**（label 恒为 0）：
      service  客服 / 催收 / 外卖坐席：高频外呼、客户池循环拨打、工作时段集中
      family   一家人共享同一部手机、家庭网络与小区基站
      growth   正常业务增长：近 10 天通话量翻数倍（换工作 / 跑销售），目标是熟人池
      event    良性突发：家长群 / 社区通知 / 红白事，截点前 1 小时打了一圈熟人电话

    正例里另加 surg（慢热转爆发）：9 天爬坡 + 末段集中拨打，用来真正触发
    「双尺度互证」——只有它同时具备短时突发与长时渐变。
    """
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)
    uid = 0
    users: Dict[int, str] = {}
    lam: Dict[int, float] = {}
    onset: Dict[int, int] = {}
    fam_of: Dict[int, int] = {}
    T = days * DAY
    TEND = (days - 1) * DAY + 21 * HOUR     # 最后一个查询截点：当地 21:00
    # 前三个沿用旧口径（21:00）；后两个是工作时段（15:00 / 11:00），
    # 用于让"合法高频外呼"在冲刺时段真正与爆发型同框，否则一律在下班后取样。
    cuts = [TEND - 7 * DAY, TEND - 2 * DAY, TEND,
            TEND - DAY - 6 * HOUR, TEND - 7 * DAY - 10 * HOUR]

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

    # ================= 难负例（label 恒为 0）=================
    # 这三类用户行为上"像诈骗"，但都是合法业务/正常生活，用来把合成集从
    # "剧本可分" 拉到真实运营商场景的难度。

    def _pick(pool, k: int):
        return [int(x) for x in npr.choice(pool, size=min(k, len(pool)), replace=False)]

    cust_pool = [CUST_BASE + i for i in range(1200)]
    lead_pool = [LEAD_BASE + i for i in range(40000)]

    # ---- 客服 / 催收 / 外卖坐席：高频外呼，工作时段「冲刺式」批量拨打 ----
    # 与爆发型的区别：集中在工作日白天、接通率高、时长较长、线索与存量客户混拨。
    # 与慢热型的区别：没有以天为单位的爬坡。
    service_users = [new_user("service", 0.0) for _ in range(n_service)]
    for u in service_users:
        pool = _pick(cust_pool, rng.randint(220, 360))
        ptr = rng.randrange(max(1, len(pool)))
        lptr = (u * 7919) % 5000                    # 每个坐席从线索池的不同位置开始
        for d in range(days):
            if (d % 7) >= 5 and rng.random() < .75:      # 周末基本不出勤
                continue
            for h0, h1, k, lead_share in ((9.3, 11.6, rng.randint(14, 22), .60),
                                          (13.8, 16.2, rng.randint(16, 26), .65),
                                          (18.2, 20.2, rng.randint(8, 14), .40)):
                if k < 1:
                    continue
                spr = h0 + rng.random() * max(0.0, h1 - h0 - 0.3)
                dense = int(k * .70)
                for i in range(k):
                    # 七成通话挤进 10 分钟，与「爆发型」的短时冲刺同形
                    t = d * DAY + int(((spr + rng.random() * .17) if i < dense
                                       else (h0 + rng.random() * (h1 - h0))) * HOUR)
                    if rng.random() < lead_share:       # 新线索：只打一次
                        b = lead_pool[lptr % len(lead_pool)]
                        lptr += 1
                    else:                               # 存量客户：循环回访
                        b = pool[ptr % len(pool)]
                        ptr += 1
                    # 预测式外呼：约两成是拨通即挂的超短呼叫，接通率也偏低
                    talk = int(rng.uniform(3, 9)) if rng.random() < .22 else int(rng.uniform(15, 115))
                    add(u, b, t, talk, CONNECT if rng.random() < .62 else UNANSWERED,
                        True, rng.randint(200, 4000))

    # ---- 夜间班次坐席：部分催收/贷款中介在 21:00–23:30 外呼（夜间主叫比例升高）----
    for u in service_users[:max(1, n_service // 4)]:
        pool = _pick(cust_pool, rng.randint(150, 260))
        ptr, lptr2 = 0, (u * 104729) % 20000
        for d in range(days):
            if rng.random() < .45:                       # 不是每晚都有班
                continue
            for _ in range(rng.randint(6, 14)):
                b = (lead_pool[lptr2 % len(lead_pool)] if rng.random() < .55
                     else pool[ptr % len(pool)])
                ptr += 1; lptr2 += 1
                t = d * DAY + int(rng.uniform(21.0, 23.4) * HOUR)
                talk = int(rng.uniform(3, 10)) if rng.random() < .3 else int(rng.uniform(12, 90))
                add(u, b, t, talk, CONNECT if rng.random() < .55 else UNANSWERED,
                    True, rng.randint(200, 4000))

    # ---- 一家人共享设备：4 名成员共用同一部手机 / 家庭网络 / 小区基站 ----
    # 对五元路径是硬负例：设备、网络、基站三条路径都会把家人连成"团伙"，但他们是正常用户。
    for fam_id in range(n_family):
        fam = [new_user("family", 0.0) for _ in range(family_size)]
        circle = _pick(OUT_POOL, rng.randint(12, 25))     # 一家人共用的亲友圈
        for m in fam:
            fam_of[m] = fam_id
        for d in range(days):
            for a in fam:
                for b in fam:
                    if a == b or rng.random() > .50:
                        continue
                    t = d * DAY + int(min(23.9, max(0.0, rng.gauss(19.8, 2.6))) * HOUR)
                    add(a, b, t, int(max(5, rng.gauss(75, 45))),
                        CONNECT if rng.random() < .85 else UNANSWERED, True, rng.randint(200, 3000))
                # 各自的外部往来：六成打给同一批亲友（共同被叫目标路径也是硬负例）
                for _ in range(int(npr.poisson(5.0))):
                    b = int(rng.choice(circle)) if rng.random() < .60 else int(rng.choice(OUT_POOL))
                    t = _diurnal(rng, d)
                    add(a, b, t, int(max(5, rng.gauss(90, 60))),
                        CONNECT if rng.random() < .72 else UNANSWERED, True, rng.randint(200, 5000))

    # ---- 正常业务增长：近 10 天通话量翻数倍，目标是熟人池（新目标比例不爆发）----
    # 对 S2 的长时漂移是硬负例：基线 3 通/天 → 期末 25 通/天，但这是换工作，不是慢热诈骗。
    growth_users = [new_user("growth", 0.0) for _ in range(n_growth)]
    for u in growth_users:
        pool = _pick(cust_pool[:400] + OUT_POOL[:300], rng.randint(60, 120))
        lptr = (u * 3571) % 20000
        for d in range(days):
            ramp = max(0, d - (days - 10))
            base = 3 if ramp == 0 and d < days - 10 else int(3 + ramp * 2.4)
            for _ in range(base):
                # 爬坡期四成打新客户（跑销售/接单），其余打熟人 —— 新目标比例升高到 0.3~0.5
                if ramp > 0 and rng.random() < .45:
                    b = lead_pool[lptr % len(lead_pool)]
                    lptr += 1
                else:
                    b = pool[rng.randrange(len(pool))]
                t = d * DAY + int(min(23.9, max(0.0, rng.gauss(19.2, 2.0))) * HOUR)
                add(u, b, t, int(max(6, rng.gauss(50, 28))),
                    CONNECT if rng.random() < .66 else UNANSWERED, True, rng.randint(200, 5000))

    # ---- 良性突发：家长群 / 社区通知 / 红白事——截点前 1 小时打了一圈电话 ----
    # 真实投诉里最主要的误报来源：短时行为与爆发型同形，但目标都是熟人、只此一次。
    event_users = [new_user("event", 0.0) for _ in range(n_event)]
    for i, u in enumerate(event_users):
        cut = cuts[i % len(cuts)]
        circle = _pick(OUT_POOL, 24)
        for d in range(days):                                  # 日常就是普通用户
            for _ in range(int(npr.poisson(6))):
                add(u, int(rng.choice(circle)), _diurnal(rng, d),
                    int(max(8, rng.gauss(70, 40))),
                    CONNECT if rng.random() < .7 else UNANSWERED, True, rng.randint(200, 5000))
        n_ev = rng.randint(26, 46)                             # 事件当天的一圈电话
        dense = int(n_ev * .65)
        for j in range(n_ev):
            t = cut - (rng.randint(2, 14) if j < dense else rng.randint(15, 75)) * MIN
            add(u, int(rng.choice(circle)), t, int(rng.uniform(8, 55)),
                CONNECT if rng.random() < .8 else UNANSWERED, True, rng.randint(200, 3000))

    # ---- 慢热转爆发（正例）：9 天爬坡 + 末段集中拨打 ----
    # 唯一同时具备「长时渐变」与「短时突发」的诈骗形态，也是说明书 2.4 双尺度互证
    # 与 claim 4/5 要处理的对象；只用独立随机源 rng3，保证既有用户的画序逐条不变。
    rng3 = random.Random(seed * 7919 + 13)
    for _ in range(n_surg):
        u = new_user("surg", .10, TEND - 9 * DAY)
        close = [int(x) for x in npr.choice([k for k in users if k != u], size=6, replace=False)]
        for d in range(days - 9):                       # 前 21 天：正常用户
            for _ in range(int(npr.poisson(6))):
                b = int(rng3.choice(close)) if rng3.random() < .6 else int(rng3.choice(OUT_POOL))
                add(u, b, _diurnal(rng3, d), int(max(8, rng3.gauss(80, 45))),
                    CONNECT if rng3.random() < .70 else UNANSWERED, True, rng3.randint(200, 5000))
        for d in range(days - 9, days):                 # 后 9 天：逐日爬坡
            k = 6 + (d - (days - 9)) * 3
            for _ in range(k):
                t = d * DAY + int(min(23.9, max(0.0, rng3.gauss(19.5, 2.0))) * HOUR)
                add(u, VICTIM_BASE + int(rng3.uniform(0, 400)), t, int(rng3.uniform(8, 50)),
                    CONNECT if rng3.random() < .65 else UNANSWERED, True, rng3.randint(200, 5000))
        for _ in range(int(rng3.randint(45, 75))):      # 末段 55 分钟：集中拨打
            t = TEND - rng3.randint(1, 55) * MIN
            add(u, VICTIM_BASE + int(rng3.uniform(0, 500)), t, int(rng3.uniform(3, 20)),
                CONNECT if rng3.random() < .80 else UNANSWERED, True, rng3.randint(100, 2500))

    # ---- 关系观测 ----
    obs: List[Obs] = []
    oid = 0

    def add_obs(u: int, kind: str, key: str, t: int, delay: int = 5000,
                etype: str = None) -> None:
        nonlocal oid
        if etype is None and kind == "P":      # 支付实体必须是可核验的账户/收款实体/支付终端
            etype = ("account", "merchant", "terminal")[oid % 3]
        obs.append(Obs(f"o{oid:07d}", u, kind, key, t, t + delay, True, etype or "account"))
        oid += 1

    def keys_for(u: int) -> Dict[str, str]:
        k = users[u]
        if k == "family":
            fid = fam_of[u]
            return {"D": f"dev-fam{fid}", "I": f"ip-fam{fid}", "B": f"bs-fam{fid}", "P": f"pay-u{u}"}
        if k == "service":
            office = u % 3                      # 同一坐席组的 3 人共用办公网络与基站
            return {"D": f"dev-svc{u}", "I": f"ip-svc{office}", "B": f"bs-svc{office}", "P": f"pay-u{u}"}
        if k == "growth":
            return {"D": f"dev-u{u}", "I": f"ip-v{u % 40}", "B": f"bs-v{u % 40}", "P": f"pay-u{u}"}
        if k == "ring":
            r = next(r for r in rings if u in r)
            return {"D": f"dev-ring{r[0]}", "I": f"ip-ring{r[0]}", "B": f"bs-ring{r[0]}", "P": f"pay-ring{r[0]}"}
        if k in ("burst", "surg"):
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

    # ---- 通话级覆盖观测：让每类关联在同一窗口内覆盖该用户的通话（口径见 s3.quality）----
    # 用独立随机源，避免改变主 rng 序列（保证 calls / queries 与旧数据集逐字节一致）
    rng2 = random.Random(seed * 1000003 + 11)
    for c in calls:
        for u in (c.caller, c.callee):
            if u not in users:          # 外部号码不在用户表内，跳过
                continue
            base = keys_for(u)
            for kd in ("D", "I", "B", "P"):
                if rng2.random() >= COVER_P:
                    continue
                key = base[kd]
                if users[u] == "burst" and c.start >= onset.get(u, T):
                    key = f"{base[kd]}-hot{(c.start // (8 * MIN)) % 3}"
                add_obs(u, kd, key, c.start, rng2.randint(300, 2500))

    # ---- 查询截点 ----
    queries: List[Query] = []
    qn = 0

    for u, kind in users.items():
        for c in cuts:
            if c <= DAY:
                continue
            label = int(kind in FRAUD_KINDS and c >= onset.get(u, T + 1))
            qc = 0.990 + rng.uniform(0, .009)
            if rng.random() < .04:
                qc = 0.94
            queries.append(Query(f"q{qn:05d}", u, c, label, round(min(.999, max(.90, qc)), 4),
                                 round(0.80 + rng.uniform(0, .18), 4), rng.random() > .01))
            qn += 1
    return Dataset(calls, obs, queries, users, onset)
