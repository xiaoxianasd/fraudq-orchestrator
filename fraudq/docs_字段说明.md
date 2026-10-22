# 批量接入数据字段说明

## 文件要求
- 编码：**UTF-8 with BOM**（Excel 才能正确显示中文）；程序读取用 utf-8-sig，两种都兼容
- 格式：`.csv` 或 `.jsonl`（每行一个 JSON 对象）
- 一行只能是「通话行」或「关联观测行」，系统按表头判定

## 通话行（必须含 caller）

| 字段 | 别名 | 必填 | 类型 | 说明 |
|---|---|---|---|---|
| call_id | cid / 编号 | 否 | str | 幂等键，重复写入会被跳过；留空自动生成 |
| caller | 主叫 | 是 | int | 主叫号码，正整数 |
| callee | 被叫 | 是 | int | 被叫号码，正整数且 ≠ caller |
| start_at | 时间 | 是 | int(ms) | 通话发生时间（毫秒时间戳） |
| talk_seconds | 时长 | 否 | int | 有效接通秒数，未接通填 0，不能为负 |
| status | 状态 | 否 | str/int | 接通/未接/忙线/失败，或 0/1/2/3，默认接通 |
| available_at | 可用时间 | 否 | int(ms) | 系统何时看到这条记录；缺省 = 结束 + 0.5s；必须 ≥ start_at |

## 关联观测行（必须含 user）

| 字段 | 别名 | 必填 | 类型 | 说明 |
|---|---|---|---|---|
| user | 用户 | 是 | int | 观测归属用户 |
| kind | 类型 | 是 | str | D 共享设备 / I 网络地址 / B 基站位置 / P 支付实体 |
| key | 标识 | 是 | str | 设备号、IP、基站号、支付实体号 |
| observed_at | 观测时间 | 是 | int(ms) | 该关联被观测到的时刻 |
| available_at | 可用时间 | 否 | int(ms) | 缺省 = 观测时间 + 1s |

## 核心口径：发生时间 vs 可用时间

- `start_at` / `observed_at` = 事情**什么时候发生**
- `available_at` = 系统**什么时候看到**这条记录（真实场景里话单有延迟、补传、迟到）
- 全系统只使用 `available_at ≤ 查询截点` 的记录，这是专利的核心约束
- 若 `available_at < start_at`，该行会被判为非法并拒绝写入

## 校验与幂等

1. 主被叫须为正整数且不相同
2. 时长 ≥ 0，状态在枚举内
3. `available_at ≥ start_at`
4. 同 `call_id` 重复写入 → 计入 duplicated，不报错

## 写入后自动执行

校验 → 写库 → 缓存失效 → 对受影响用户自动重判

- 只重判**主叫**，以及插入前就已存在的**被叫**
- 历史不足 0.05 天的全新用户跳过（没有画像，判了也没意义）

## 示例

```csv
call_id,caller,callee,start_at,available_at,talk_seconds,status,user,kind,key,observed_at
feed-100-0,100,950000,2577600000,2577612000,9,接通,,,,
obs-100-0,,,,,,,100,D,dev-u100,2577600000
```