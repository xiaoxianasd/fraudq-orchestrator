"""cli — 命令行交互界面：启动 logo + REPL。"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter

from .core import REL_KINDS, REL_NAME, STAT_NAMES

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.system("")  # 开启 Windows 控制台 VT 序列

C = dict(r="\033[31m", g="\033[32m", y="\033[33m", b="\033[34m", m="\033[35m",
         c="\033[36m", w="\033[37m", d="\033[2m", B="\033[1m", x="\033[0m")

STATUS_COLOR = {"高危": "r", "快通道告警": "r", "高危(敏感)": "m", "可疑": "y",
                "低危": "g", "信息不足": "d"}
ACTION_COLOR = {"阻断": "r", "预警": "m", "观察": "y", "放行": "g", "无": "d"}

LOGO = r"""
   ______ ______   ___  __  ______  __  ____
  / ____// __  /  /   |/ / / __  / / / / __ \
 / /_   / /_/ /  / /| |/ / / / / / / / / /_/ /
/ __/  / _, _/  / / |  / / /_/ / /_/ / _, _/
/_/   /_/ |_|  /_/  |_/  \____/ \____//_/ |_|
"""


import unicodedata as _ud


def dw(s: str) -> int:
    """按终端显示宽度计算（CJK 记 2 列）。"""
    return sum(2 if _ud.east_asian_width(ch) in "WF" else 1 for ch in s)


def c(text: str, *styles: str) -> str:
    return "".join(C[s] for s in styles) + text + C["x"]


def bar(v: float, width: int = 22) -> str:
    n = int(max(0.0, min(1.0, v)) * width)
    col = "r" if v >= .8 else ("y" if v >= .2 else "g")
    return c("█" * n, col) + c("·" * (width - n), "d")


def banner(ds_stats: dict, model: str) -> None:
    w = 78
    print(c(LOGO.rstrip("\n"), "c", "B"))
    print(c("  多智能体协同反诈电话检测 · 终端", "B") + c("   v1.0", "d"))
    print(c("  " + "─" * w, "d"))
    s = ds_stats
    print(c(f"  数据 {s['users']} 用户 / {s['calls']} 通话 / {s['obs']} 关联观测 / "
            f"{s['queries']} 查询（正例 {s['positive']}）", "d"))
    print(c(f"  模型 {os.path.basename(model)}   输入项 :help 查看命令，:q <用户号> 发起一次研判", "d"))
    print(c("  " + "─" * w, "d"))


class App:
    def __init__(self, ds, eng, coord, ds_path: str, model_path: str):
        self.ds = ds
        self.eng = eng
        self.coord = coord
        self.ds_path = ds_path
        self.model_path = model_path
        self.by_user = {}
        for q in ds.queries:
            self.by_user.setdefault(q.user, []).append(q)
        for v in self.by_user.values():
            v.sort(key=lambda q: q.cutoff)
        self.last_trace = None
        self.budget = 200.0
        self.verbose = True
        self._n = 0


    # ---------- 过程旁白 ----------
    STAGE_COLOR = {"data": "c", "dq": "y", "fast": "m", "screen": "b", "S1": "g", "S2": "y",
                   "HAN": "m", "trigger": "d", "fuse": "b", "replay": "y", "judge": "w"}

    def sink(self, stage: str, msg: str, kw: dict) -> None:
        self._n += 1
        col = self.STAGE_COLOR.get(stage, "w")
        head = c(f" [{self._n}] ", "d") + c(f"{stage:<7}", col, "B")
        body = []
        if stage == "data":
            rel = " ".join(f"{k}{v}" for k, v in (kw.get("rels") or {}).items() if v)
            body.append(f"可见记录 {kw.get('visible')} 条 / 有效掩码 {kw.get('mask')}/16 / 历史 {kw.get('history_days')} 天")
            body.append(f"关系邻居 {rel or '无'}")
        elif stage == "fast":
            tag = c("命中", "r", "B") if kw.get("hit") else c("未命中", "d")
            body.append(f"10 分钟主叫 {kw.get('calls')} 通 / 新目标 {kw.get('new_ratio')}  →  " + tag)
            body.append(c(f"规则 {kw.get('rule')}", "d"))
        elif stage == "dq":
            lv = kw.get("level")
            col = "y" if lv == "degraded" else "r"
            print(head + "  " + c(f"话单完整性 {kw.get('q_call')} → {lv}", col) +
                  c(f"   {kw.get('ceiling')}", "d"))
            return
        elif stage == "screen":
            body.append(f"p0={kw.get('p0')}  u={kw.get('u')}  G_u={int(bool(kw.get('g_u')))}")
        elif stage == "skip":
            print(head + "  " + c(f"{kw.get('agent')} 未执行（{kw.get('reason')}）", "d"))
            return
        elif stage == "S1":
            body.append(f"p_T={kw.get('p_T', '')}")
            body.append(c(f"C_sem={kw.get('c_sem')} 检查点={kw.get('checkpoints')}", "d"))
            body.append(c(f"→ {msg}   {kw.get('verdict')}   {kw.get('ms')}ms", "g"))
        elif stage == "S2":
            body.append(f"B_s={kw.get('B_s')} B_l={kw.get('B_l')} C_temp={kw.get('C_temp')} "
                        f"λ={kw.get('lam')} E_ts={kw.get('E_ts')}")
            body.append(c(f"窗口 W_s={kw.get('ws')}min W_l={kw.get('wl')}d   {kw.get('ms')}ms", "d"))
        elif stage == "HAN":
            pt = " ".join(f"{k}:{v}" for k, v in (kw.get("paths") or {}).items())
            body.append(f"p_H={kw.get('p_H', '')}  路径权重 {pt or '—'}")
            body.append(c(f"→ {msg}   {kw.get('verdict')}   {kw.get('ms')}ms", "m"))
        elif stage == "trigger":
            body.append(msg + c(f"   ({kw.get('reason', '')})", "d"))
        elif stage == "fuse":
            body.append(f"p={kw.get('p')}  分支差={kw.get('gap')}")
        elif stage == "replay":
            if kw.get("mode") == "async":
                body.append(c(msg + "  →  在线路径结束，复核转后台", "y"))
            else:
                body.append(msg)
        elif stage == "judge":
            vc = {"疑似诈骗": "r", "需观察": "y", "安全": "g"}.get(kw.get("verdict"), "w")
            body.append(c(f"结论 {msg}", vc, "B") + c(f"   处置 {kw.get('action')}   "
                        f"在线 {kw.get('ms')}ms", "d"))
        else:
            body.append(msg)
        line = head + "  " + body[0]
        print(line)
        for extra in body[1:]:
            print(" " * 9 + extra)

    # ---------- 渲染 ----------
    def report(self, tr) -> None:
        r = tr.results
        s1, s2, han, fu, rp = r.get("S1", {}), r.get("S2", {}), r.get("HAN", {}), r.get("FUSE", {}), r.get("REPLAY", {})
        q = next((x for x in self.ds.queries if x.qid == tr.qid), None)
        print()
        print(c(f"┌ 查询 {tr.qid}   用户 {tr.user}   截点 t+{tr.cutoff/86400000:.2f}d ", "B"))
        p_show = fu.get("p", r.get("SCREEN", {}).get("p0", 0.0))
        print(c("│", "d") + f" 初筛    p0={r.get('SCREEN',{}).get('p0',0):.3f}  综合证据   " + bar(p_show))
        if s1:
            print(c("│", "d") + f" S1      p_T={s1['p']:.3f}  p^_T={s1['p_cal']:.3f}  "
                  f"C_sem={s1['c_sem']:.2f}  检查点={s1['checkpoints']}")
            top = "、".join(f"{STAT_NAMES[i]}({v})" for i, v in s1.get("top", []))
            print(c("│", "d") + c(f"         Top 特征：{top}", "d"))
        if s2:
            print(c("│", "d") + f" S2      B_s={s2['B_s']:.2f}  B_l={s2['B_l']:.2f}  "
                  f"C_temp={s2['C_temp']:.2f}  λ={s2['lam']}  E_ts={s2['E_ts']:.3f}  [{s2['flag']}]")
            print(c("│", "d") + c(f"         窗口 W_s={s2['W_s']/60000:.0f}min W_l={s2['W_l']/86400000:.0f}d  "
                  f"K={s2['K']} D={s2['D']}  复现度R={s2['R']:.2f}", "d"))
        if han:
            beta = han.get("beta") or [0] * 5
            bs = "  ".join(f"{k}:{b:.2f}" for k, b in zip(REL_KINDS, beta) if b > 1e-3)
            print(c("│", "d") + f" HAN     p^_H={han['p_cal']:.3f}  路径权重 {bs or '无非零路径'}")
        if fu:
            print(c("│", "d") + f" 融合    p={fu['p']:.3f}   " + bar(fu["p"]))
        if rp and rp.get("scenes"):
            parts = "  ".join(f"{k}={v['delta']:+.3f}" for k, v in rp["scenes"].items())
            print(c("│", "d") + f" 重放    Δ_g {parts or '—'}")
        elif rp and rp.get("pending"):
            print(c("│", "d") + c(" 复核    已挂起（异步），在线路径不等待", "y"))
        col = STATUS_COLOR.get(tr.status, "w")
        acol = ACTION_COLOR.get(tr.action, "w")
        rng = ""
        if fu and "p_lo" in fu:
            rng = c(f"  区间[{fu['p_lo']:.3f}, {fu['p_hi']:.3f}]", "d")
        print(c("├", "d") + c(f" 判断  {tr.status}", col, "B") +
              c(f"   决策 {tr.action}", acol, "B") + rng +
              f"   在线 {tr.elapsed_ms:.1f}ms")
        print(c("└", "d") + c(" 输入 :why 看调度决策树，:replay 看证据组", "d"))

    def why(self) -> None:
        if not self.last_trace:
            print(c("  还没有查询记录，先 :q <用户号>", "y")); return
        print()
        print(c("  调度决策树（证据驱动的动态插入）", "B"))
        for i, d in enumerate(self.last_trace.decisions, 1):
            mark = {"run": "→", "跳过": "×", "降级跳过": "!"}.get(d["choice"], "·")
            col = "g" if d["choice"] == "run" else ("r" if d["choice"] == "降级跳过" else "d")
            print(f"  {i:>2}. " + c(f"[{d['rule']}]", "c") + f" 条件 {d['cond']}  " +
                  c(f"{mark} {d['choice']}", col) + f"  {d['cost_ms']:.1f}ms")
            if d["reason"]:
                print(c(f"       └ {d['reason']}", "d"))
        print()

    def replay(self) -> None:
        rp = (self.last_trace or type("x", (), {"results": {}})()).results.get("REPLAY")
        if not rp or not rp.get("scenes"):
            print(c("  本次未触发隔离重放（p<0.80 且分支差<0.30）", "d")); return
        print()
        print(c(f"  原始证据隔离重放   完整 logit={rp['p_full']:.3f}（Δ 以对数几率计，避免概率饱和）", "B"))
        for k, v in rp["scenes"].items():
            sign = "+" if v["delta"] > 0 else ""
            print(f"  {k:<9} 移除 {v['n']:>3} 条/键  p_without={v['p_without']:.3f}  " +
                  c(f"Δ={sign}{v['delta']:.3f}", "r" if v["delta"] >= .1 else "d"))
        print(c("  Δ>0.10 表示该组原始记录直接支撑结论，可作人工核查依据", "d"))
        print()

    def agents(self) -> None:
        print()
        print(c("  智能体注册表", "B"))
        rows = [("SCREEN", "轻量初筛 44→16→2 头", 754),
                ("S1", "行为语义 TGN 记忆", sum(p.numel() for p in self.eng.m["tgn"].parameters())),
                ("S2", "双尺度时序核验", "解析"),
                ("HAN", "五元路径图注意力", sum(p.numel() for p in self.eng.m["han"].parameters())),
                ("FUSE", "15 维可信度融合", sum(p.numel() for p in self.eng.m["fusion"].parameters())),
                ("REPLAY", "检查点隔离重放", "解析")]
        for name, desc, n in rows:
            ms = self.eng._cost.get(name)
            cost = f"{ms:5.1f}ms" if ms else "  —  "
            print(f"  {c(name.ljust(7), 'c')} {desc.ljust(22)} 参数 {str(n).rjust(6)}   实测 {cost}")
        print(c("  调度器按 eligible → 预算 → 触发条件 决定是否执行；未执行分支以掩码位进入融合", "d"))
        print()

    def data(self) -> None:
        s = self.ds.stats()
        print()
        print(c("  数据集", "B") + c(f"  {self.ds_path}", "d"))
        print(f"  用户 {s['users']}  通话 {s['calls']}  关联观测 {s['obs']}  查询 {s['queries']}"
              f"（正 {s['positive']} / 负 {s['negatives']}）")
        print(f"  欺诈用户 {s['fraud_users']}  观测构成 " +
              " ".join(f"{k}:{v}" for k, v in s["kinds"].items()))
        print(c("  字段口径：available_at ≤ cutoff 才可见；event_time 与 available_at 分离", "d"))
        print()

    def query(self, user: int, budget: float | None = None, cut: int | None = None) -> None:
        qs = self.by_user.get(user)
        if not qs:
            print(c(f"  用户 {user} 没有查询截点（范围 0..{max(self.by_user)})", "r")); return
        q = qs[cut] if cut is not None and -len(qs) <= cut < len(qs) else qs[-1]
        self._n = 0
        tr = self.coord.run(q, budget or self.budget, replay_mode="defer",
                            sink=self.sink if self.verbose else None)
        online = tr.elapsed_ms
        self.last_trace = tr
        self.report(tr)
        if tr.results.get("REPLAY", {}).get("pending"):
            tr, rms = self.coord.finalize(tr)
            self.last_trace = tr
            print(c(f" [{self._n + 1}] ", "d") + c(f"{'REPLAY':<7}", "y", "B") + "  " +
                  c(f"异步复核完成 +{rms:.0f}ms（不占在线时延）", "d"))
            for k, v in tr.results["REPLAY"]["scenes"].items():
                sign = "+" if v["delta"] > 0 else ""
                print(" " * 9 + f"{k:<8} 移除 {v['n']:>3} 条/键  p_without={v['p_without']:.3f}  " +
                      c(f"Δ={sign}{v['delta']:.3f}", "r" if v["delta"] >= .5 else "d"))
            print(" " * 9 + c(f"→ 最终状态 {tr.status} / {tr.action}", "w", "B"))
        if q.label >= 0:
            hit = (tr.results.get("FUSE", {}).get("p", 0) >= .20) == bool(q.label)
            print(c(f"  真值 {q.label} · 判定{'一致' if hit else '不一致'}", "d"))

    def probe(self, user: int, minutes: float) -> None:
        """在任意时点发起研判（相对该用户最后一个截点偏移 N 分钟），用于演示快通道。"""
        from .core import Query
        qs = self.by_user.get(user)
        if not qs:
            print(c(f"  用户 {user} 无历史截点", "r")); return
        base = qs[-1]
        on = self.ds.onset.get(user, 10 ** 18)
        if str(minutes).startswith("on+"):
            minutes = (on + float(str(minutes)[3:]) * 60000 - base.cutoff) / 60000
        t = int(base.cutoff + minutes * 60000)
        kind = self.ds.archetype.get(user, "normal")
        label = int(kind != "normal" and t >= on)
        q = Query(f"probe-{user}-{int(minutes)}", user, t, label, .99, .9, True)
        self._n = 0
        if t not in self.eng._peer:
            print(c("  首次在该截点推理，预热记忆快照（约 1-3 秒）…", "d"))
        print(c(f"  探针：用户 {user}（真值形态 {kind}），截点 = 最后截点 {minutes:+.0f} 分钟", "d"))
        tr = self.coord.run(q, self.budget, replay_mode="defer", sink=self.sink if self.verbose else None)
        self.last_trace = tr
        self.report(tr)

    # ---------- 在线写入 ----------
    def add_call(self, caller: int, callee: int, talk: int = 30, status_name: str = "接通") -> None:
        from .core import Call, CONNECT, UNANSWERED, BUSY, FAILED
        smap = {"接通": CONNECT, "未接": UNANSWERED, "忙线": BUSY, "失败": FAILED}
        st = smap.get(status_name, CONNECT)
        now = max(self.eng.store.latest(caller), self.eng.store.latest(callee), 0)
        t = now + 60_000
        n = len(self.eng.store.calls)
        c_ = Call(f"live{n:06d}", caller, callee, t, t + talk * 1000,
                  t + talk * 1000, talk, st, True, True, True)
        ok = self.eng.store.add_call(c_)
        if ok:
            self.eng.invalidate(t)
        print(c("  " + ("已写入" if ok else "重复，已忽略") +
                f" 通话 {c_.cid}: {caller} → {callee}  {talk}s  {status_name}  "
                f"t+{t/86400000:.3f}d", "g" if ok else "y"))
        print(c(f"  共 {len(self.eng.store.calls)} 条通话；用 :now {caller} 立刻重判", "d"))

    def now(self, user: int) -> None:
        from .core import Query
        t = self.eng.store.latest(user)
        if t <= 0:
            print(c(f"  用户 {user} 还没有任何记录", "r")); return
        q = Query(f"live-{user}-{len(self.ds.queries)}", user, t, -1, .99, .95, True)
        self._n = 0
        print(c(f"  在线研判：用户 {user}  截点 = 最新可见时刻 t+{t/86400000:.3f}d", "d"))
        tr = self.coord.run(q, self.budget, replay_mode="defer",
                            sink=self.sink if self.verbose else None)
        self.last_trace = tr
        self.report(tr)

    def batch(self, n: int = 60) -> None:
        pool = [q for q in self.ds.queries]
        random.Random(0).shuffle(pool)
        pool = pool[:n]
        from .runtime import batch
        print(c(f"  批量评估 {len(pool)} 条查询 …", "d"))
        res = batch(self.coord, pool, self.budget)
        print()
        print(c("  指标", "B"))
        print(f"  AUC {res['auc']:.3f}   F1 {res['f1']:.3f}   精确率 {res['precision']:.3f}   "
              f"召回率 {res['recall']:.3f}")
        print(f"  时延 P50 {res['p50']:.1f}ms  P95 {res['p95']:.1f}ms   总耗时 {res['total_s']:.1f}s")
        print(c("  风险状态分布", "B"))
        for k, v in res["status"].most_common():
            print("   " + c(f"{k:<12}", STATUS_COLOR.get(k, "w")) + " " + bar(v / len(pool), 30) + f" {v}")
        print()
        self.last_batch = res

    # ---------- REPL ----------
    def loop(self) -> None:
        print()
        while True:
            try:
                line = input(c("fraudq› ", "c", "B")).strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not line:
                continue
            self.dispatch(line)

    COMMANDS = {":q", ":query", ":why", ":replay", ":probe", ":batch", ":agents",
                ":data", ":step", ":budget", ":help", ":quit", ":add", ":now", ":feed"}

    def dispatch(self, line: str) -> bool:
        parts = line.split()
        if not parts:
            return True
        # 允许不带冒号（q / step on / help ...）
        if not parts[0].startswith(":") and (":" + parts[0]) in self.COMMANDS:
            parts[0] = ":" + parts[0]
        cmd, args = parts[0], parts[1:]
        if cmd in (":q", ":query"):
            if not args:
                print(c("  用法 :q <用户号> [预算ms] [截点序号]", "y")); return True
            self.query(int(args[0]), float(args[1]) if len(args) > 1 else None,
                       int(args[2]) if len(args) > 2 else None)
        elif cmd == ":why":
            self.why()
        elif cmd == ":replay":
            self.replay()
        elif cmd == ":agents":
            self.agents()
        elif cmd == ":data":
            self.data()
        elif cmd == ":probe":
            if len(args) < 2:
                print(c("  用法 :probe <用户> <相对分钟数，可为负>", "y")); return True
            self.probe(int(args[0]), args[1] if args[1].startswith("on+") else float(args[1]))
        elif cmd == ":add":
            if len(args) < 2:
                print(c("  用法 :add <主叫> <被叫> [秒数] [接通|未接|忙线|失败]", "y")); return True
            self.add_call(int(args[0]), int(args[1]),
                          int(args[2]) if len(args) > 2 else 30,
                          args[3] if len(args) > 3 else "接通")
        elif cmd == ":feed":
            if not args:
                print(c("  用法 :feed <calls.csv|.jsonl>", "y")); return True
            from .ingest import ingest_csv
            try:
                rep = ingest_csv(self.eng.store, self.eng, args[0], coord=self.coord)
            except FileNotFoundError:
                print(c(f"  文件不存在：{args[0]}", "r")); return True
            print(c("  批量接入报告", "B") +
                  f"  写入通话 {rep.calls} / 观测 {rep.obs} / 重复 {rep.duplicated} / 拒绝 {len(rep.invalid)}")
            for bad in rep.invalid[:5]:
                print(c(f"    第 {bad['row']} 行：{bad['error']}", "y"))
            if rep.users:
                print(c(f"  受影响用户 {len(rep.users)} 个，自动重判：", "d"))
                for j in rep.judgments[:12]:
                    col = STATUS_COLOR.get(j["grade"], "w")
                    print(f"    u{j['user']:<6} p={j['p']:.3f}  " + c(f"{j['grade']} → {j['action']}", col) +
                          c(f"  {j['ms']}ms", "d"))
        elif cmd == ":now":
            if not args:
                print(c("  用法 :now <用户号>", "y")); return True
            self.now(int(args[0]))
        elif cmd == ":batch":
            self.batch(int(args[0]) if args else 60)
        elif cmd == ":step":
            self.verbose = (args[0] not in ("off", "0")) if args else (not self.verbose)
            print(c(f"  过程旁白 {'开启' if self.verbose else '关闭'}", "d"))
        elif cmd == ":budget":
            self.budget = float(args[0]); print(c(f"  预算 = {self.budget:.0f}ms", "d"))
        elif cmd in (":help", "?"):
            help_text()
        elif cmd in (":quit", ":q!", "exit", "quit"):
            return False
        else:
            hint = f"，是不是想输入 {':' + cmd}？" if (":" + cmd) in self.COMMANDS else ""
            print(c(f"  未知命令 {cmd}{hint}  输入 :help 查看全部命令", "y"))
        return True


def help_text() -> None:
    rows = [(":q <用户> [预算ms] [截点]", "发起一次多智能体研判"),
            (":why", "查看调度决策树（动态插入点）"),
            (":replay", "查看原始证据组与 Δ_g"),
            (":probe <用户> <分钟|on+K>", "任意时点探针；on+K = 该用户起点后 K 分钟"),
            (":add <主叫> <被叫> [秒] [状态]", "在线插入一条新通话并立即入库"),
            (":now <用户>", "以最新可见时刻为截点重新研判"),
            (":feed <文件.csv>", "批量接入并自动重判受影响用户"),
            (":batch [n]", "批量评估并打印 AUC/F1/时延"),
            (":agents", "智能体清单、参数量与实测时延"),
            (":data", "数据集与可见性口径"),
            (":step [on|off]", "开关过程旁白（默认开）"),
            (":budget <ms>", "设置默认时延预算"),
            (":help", "本帮助"),
            (":quit", "退出")]
    w = max(dw(k) for k, _ in rows) + 2
    print()
    print(c("  命令", "B") + c("（冒号可省略，如 step on）", "d"))
    print(c("  " + "─" * (w + 38), "d"))
    for k, v in rows:
        print("  " + c(k, "c") + " " * max(1, w - dw(k)) + c(v, "d"))
    print(c("  " + "─" * (w + 38), "d"))
    print()


def ensure_assets(data_dir: str, model: str, auto: bool = True):
    from .data import Dataset, generate
    from .runtime import load
    from .train import fit
    if not os.path.exists(os.path.join(data_dir, "calls.csv")):
        if not auto:
            raise SystemExit("缺少数据集，先运行 python -m fraudq --train")
        print(c("  未找到数据集，生成合成 CDR …", "y"))
        ds = generate(seed=7)
        ds.save(data_dir)
    ds = Dataset.load(data_dir)
    if not os.path.exists(model):
        if not auto:
            raise SystemExit("缺少模型，先运行 python -m fraudq --train")
        print(c("  未找到模型，开始训练（约 2-3 分钟）…", "y"))
        fit(ds, epochs=3, out=model)
    eng, coord = load(model, ds)
    return ds, eng, coord


def main(argv=None) -> None:
    ap = argparse.ArgumentParser("fraudq")
    ap.add_argument("--data", default="artifacts/data")
    ap.add_argument("--model", default="artifacts/model.pt")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--budget", type=float, default=200.0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--no-logo", action="store_true")
    a = ap.parse_args(argv)

    if a.train:
        from .data import Dataset, generate
        from .train import fit
        if not os.path.exists(os.path.join(a.data, "calls.csv")):
            generate(seed=7).save(a.data)
        ds = Dataset.load(a.data)
        fit(ds, epochs=3, out=a.model)
        return

    ds, eng, coord = ensure_assets(a.data, a.model)
    try:                                  # 一次性预热（torch 首次前向 / 记忆快照）
        t0 = time.perf_counter()
        coord.run(ds.queries[0], 5000.0, replay_mode="defer")
        print(c(f"  模型预热完成 {time.perf_counter()-t0:.2f}s", "d"))
    except Exception:
        pass
    app = App(ds, eng, coord, a.data, a.model)
    app.budget = a.budget
    if not a.no_logo:
        banner(ds.stats(), a.model)
    print(c(f"  就绪 · {len(ds.queries)} 条查询 · 预算 {app.budget:.0f}ms", "g"))
    if a.batch:
        app.batch(a.batch)
        return
    app.loop()