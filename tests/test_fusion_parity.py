# -*- coding: utf-8 -*-
"""训练口径 == 推理口径（回归测试）。

历史 bug：train.branch_logits 不写 RELQ、且见路径就跑 HAN，导致融合层的
18 维输入在训练与推理两个分布之间漂移（实测 HAN 未执行的查询里，
v[6..9] 训练全 0、线上 0.95+；约 35% 的查询 HAN 执行与否不一致）。

本测试直接调用 train.fusion_inputs（训练真正走的代码路径），
与 Coordinator.run 产出的 18 维向量逐维比对。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from fraudq.data import Dataset  # noqa: E402
from fraudq.runtime import load  # noqa: E402
from fraudq.scheduler import Coordinator  # noqa: E402
from fraudq.train import fusion_inputs  # noqa: E402

DATA = os.path.join("artifacts", "data")
MODEL = os.path.join("artifacts", "model.pt")


def _assets():
    if not (os.path.exists(os.path.join(DATA, "calls.csv")) and os.path.exists(MODEL)):
        return None
    ds = Dataset.load(DATA)
    eng, coord = load(MODEL, ds)
    return ds, eng, coord


def test_fusion_input_matches_scheduler():
    assets = _assets()
    if assets is None:
        print("  skip: 缺少 artifacts/（先跑 python -m fraudq --train）")
        return
    ds, eng, coord = assets
    checked = 0
    for q in ds.queries[:40]:
        tr = coord.run(q, 200.0, replay_mode="defer")
        if "FUSE" not in tr.results:
            continue                                   # 被门控提前放行，没有融合输入
        v_infer = np.array(tr.results["FUSE"]["v"], dtype=float)
        c = fusion_inputs(eng, coord.s1, coord.s2, coord.han, eng.m["fusion"], q)
        v_train = np.array(c.get("FUSE")["v"], dtype=float)
        assert np.allclose(v_train, v_infer, atol=1e-5), (
            "训练/推理融合输入不一致: %s\n训练=%s\n推理=%s" % (
                q.qid, np.round(v_train, 3), np.round(v_infer, 3)))
        checked += 1
    assert checked > 0, "没有可比对的查询"
    print("  比对通过 %d 条查询" % checked)


if __name__ == "__main__":
    test_fusion_input_matches_scheduler()
    print("全部通过")
