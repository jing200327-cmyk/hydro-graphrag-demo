## 可视化测评结果

![Eval Cases](https://img.shields.io/badge/MVP%20Eval-50%20cases-blue) ![Route Accuracy](https://img.shields.io/badge/Route%20Accuracy-100%25-brightgreen) ![Fact Top1](https://img.shields.io/badge/Fact%20Top1-100%25-brightgreen) ![Non Fact Keyword Hit](https://img.shields.io/badge/Non--fact%20Keyword%20Hit@10-93.3%25-brightgreen) ![Fallback](https://img.shields.io/badge/Unanswerable%20Fallback-100%25-brightgreen)

测评集共 50 条，覆盖事实查询、透水等级、因果解释、多条件查询和不可回答问题。v2 版本事实查询链路已经稳定，事实类 Top1 命中率保持 100%；主要短板集中在非事实查询关键词命中和不可回答问题兜底。v3 通过结构化路由、岩性类别精确检索、水文特征精确检索和兜底策略优化，将 non_fact keyword_hit_top10 从 50.0% 提升至 93.3%，不可回答问题兜底触发率从 20.0% 提升至 100.0%，Badcase 从 20 条降至 0 条。

### 测评集构成

| 类型 | 数量 |
| --- | --- |
| 事实查询 | 15 |
| 透水等级 | 10 |
| 因果解释 | 10 |
| 多条件查询 | 10 |
| 不可回答 | 5 |

### 整体指标

| 指标 | v2 | v3 | 变化 |
| --- | --- | --- | --- |
| 测评样本数 | 50 | 50 | 持平 |
| 成功执行数 | 50 | 50 | 持平 |
| 整体路由准确率 | 100.0% | 100.0% | +0.0 pp |
| 事实查询 Top1 命中率 | 100.0% | 100.0% | +0.0 pp |
| 非事实查询 keyword_hit_top10 | 50.0% | 93.3% | +43.3 pp |
| 不可回答兜底触发率 | 20.0% | 100.0% | +80.0 pp |
| Badcase 数量 | 20 | 0 | -20 |
| 平均置信度 | 86.6 | 88.1 | +1.54 |
| 平均耗时 | 0.246s | 0.195s | -0.050s |

### 分类测评结果

| 类型 | 数量 | v2 Keyword Hit@10 | v3 Keyword Hit@10 | v2 兜底触发 | v3 兜底触发 | v3 路由准确率 |
| --- | --- | --- | --- | --- | --- | --- |
| 事实查询 | 15 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| 透水等级 | 10 | 50.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| 因果解释 | 10 | 0.0% | 80.0% | 0.0% | 30.0% | 100.0% |
| 多条件查询 | 10 | 100.0% | 100.0% | 0.0% | 20.0% | 100.0% |
| 不可回答 | 5 | 0.0% | 0.0% | 20.0% | 100.0% | - |

### 图表

![v2/v3 overall metrics](docs/eval_results/eval_overview_v2_v3.png)

![v2/v3 category keyword hit](docs/eval_results/eval_by_type_keyword_hit.png)

### Badcase 摘要

| Badcase 原因 | v2 | v3 |
| --- | --- | --- |
| non_fact_keyword_miss_top10 | 15 | 0 |
| fallback_not_triggered | 4 | 0 |
| empty_effective_rerank | 1 | 0 |

v2 的 Badcase 主要来自非事实查询的关键词命中不足和不可回答问题未触发兜底；v3 中不可回答问题不再进入无依据检索链路，非事实查询通过结构化证据路由补强，当前测评集中 Badcase 为 0。

统计脚本：`python scripts/summarize_eval_reports.py`
