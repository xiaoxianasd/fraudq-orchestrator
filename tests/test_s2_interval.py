# -*- coding: utf-8 -*-
"""S2 短时异常区间的端点几何（回归测试）。

历史 bug：端点写成 (end, start)，导致区间时长为负，rs_=max(0,负)=0，
时间一致性 C_temp 恒为 0，「双尺度互证」(λ=1.4) 永远不会触发。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraudq.s2 import bucket_span  # noqa: E402

DAY, HOUR, MIN = 86_400_000, 3_600_000, 60_000


def test_single_bucket_span():
    t, K, T_B = 10 * DAY, 24, 5 * MIN
    for k in range(K):
        s, e = bucket_span(t, K, T_B, k, k)
        assert s < e, f"桶 {k} 区间倒挂: {s} >= {e}"
        assert e - s == T_B


def test_multi_bucket_span_length():
    t, K, T_B = 10 * DAY, 24, 5 * MIN
    s, e = bucket_span(t, K, T_B, 3, 7)
    assert e - s == 5 * T_B, "区间长度应等于桶数 × 桶长"


def test_span_within_window():
    """区间必须落在 [t-K·T_B, t] 之内，且最后一个桶右端正好是截点。"""
    t, K, T_B = 10 * DAY, 24, 5 * MIN
    s, e = bucket_span(t, K, T_B, 0, K - 1)
    assert s == t - K * T_B
    assert e == t


def test_time_consistency_reachable():
    """区间合法时，C_temp 的归一化比例必须能为正（防止再次被 max(0,·) 吃掉）。"""
    t, K, T_B, ws = 10 * DAY, 24, 5 * MIN, 2 * HOUR
    s, e = bucket_span(t, K, T_B, 20, 23)
    ratio = max(0.0, (e - s) / ws)
    assert ratio > 0 and ratio <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("全部通过")
