# fraudq — 基于多智能体协同的反诈电话检测（专利定稿实现）

零配置可跑：只依赖 torch + numpy。

## 运行

    python -m fraudq                # 启动终端（logo + 交互）
    python -m fraudq --batch 100    # 直接批量评估
    python -m fraudq --train        # 重新生成数据并训练
    python -m fraudq --budget 300   # 调大时延预算

首次运行若缺 artifacts/data 或 artifacts/model.pt，会自动生成合成 CDR 并训练（约 2 分钟）。

## 架构

| 文件 | 职责 |
|---|---|
| core.py | 事件模型、截点可见性存储(available_at <= t)、16 项统计量、预算/黑板 |
| data.py | 合成 CDR（爆发/慢热/团伙 + 正常）、CSV 存取 |
| s1.py | 行为语义智能体：32 维画像 + 128 维 TGN 记忆 + 4 头注意力 + 防污染门控 |
| s2.py | 时序模式核验：事件率反比双尺度窗口 + EWMA/CUSUM + 固定基线漂移 + 双模式库 + 时间一致性 |
| s3.py | 五元路径 HAN（路径质量门控软归一化）+ 15 维可信度融合 |
| replay.py | 6 个原始证据组 + 检查点隔离重放 + Δ_g |
| scheduler.py | 调度器（创新点）：动态任务插入 + 预算降级 + 7 级状态机 |
| train.py | 分阶段训练与工件打包 |
| cli.py | 终端界面 |

## 调度器

Stage0 754 参数初筛 -> p0, u, G_u
Stage1 取证 S1/S2（并行；预算不足按 conf/cost 贪心）
Stage2 replan：S1 的 p_T 触发 HAN；HAN 与 S2 的分支差触发 REPLAY
Stage3 15 维融合 -> 7 级状态 -> 至多一次处置

未执行分支以掩码位进入融合，绝不按零风险处理；同截点重放 <=1 次、处置 <=1 次。

## 命令

    :q <用户> [预算ms] [截点]   单次研判
    :why                        调度决策树
    :replay                     证据组与 Δ_g
    :batch [n]                  AUC/F1/时延
    :agents                     智能体清单与实测时延
    :data                       数据集