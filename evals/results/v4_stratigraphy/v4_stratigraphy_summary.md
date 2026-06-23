# v4 Stratigraphy Evaluation Summary

## 新增 10 条 Stratigraphy 测试问题

| case_id | 原问题 | 新问题 | 地层维度 | 地层名称 | expected_route | ground truth 数量 |
| --- | --- | --- | --- | --- | --- | --- |
| qa_001 | 钻孔 CHGC001 的分层 CHGC001_2 位于哪个深度范围？岩性大类和岩性小类分别是什么？ | 列出晚更新世冲洪积层对应的钻孔及分层信息。 | strat_member | 晚更新世冲洪积层 | stratigraphy_exact_retrieval | 6 |
| qa_002 | 钻孔 CHGC002 的分层 CHGC002_2 位于哪个深度范围？岩性大类和岩性小类分别是什么？ | 灯笼沙段有哪些钻孔？请给出对应层位和分层信息。 | strat_member | 灯笼沙段 | stratigraphy_exact_retrieval | 60 |
| qa_003 | 钻孔 CHGC003 的分层 CHGC003_2 位于哪个深度范围？岩性大类和岩性小类分别是什么？ | 给出人工填土在各钻孔中的分布情况和层位信息。 | strat_member | 人工填土 | stratigraphy_exact_retrieval | 53 |
| qa_004 | 钻孔 CHGC004 的分层 CHGC004_2 位于哪个深度范围？岩性大类和岩性小类分别是什么？ | 列出光明村层相关的所有钻孔、层位和分层信息。 | strat_member | 光明村层 | stratigraphy_exact_retrieval | 10 |
| qa_005 | 钻孔 CHGC005 的分层 CHGC005_3 位于哪个深度范围？岩性大类和岩性小类分别是什么？ | 给出晚更新世冲洪积层在各钻孔中的分布情况和层位信息。 | strat_group | 晚更新世冲洪积层 | stratigraphy_exact_retrieval | 6 |
| qa_006 | 钻孔 CHGC006 的分层 CHGC006_2 位于哪个深度范围？岩性大类和岩性小类分别是什么？ | 列出人工填土对应的钻孔及分层信息。 | strat_group | 人工填土 | stratigraphy_exact_retrieval | 53 |
| qa_007 | 钻孔 CHGC007 的分层 CHGC007_2 位于哪个深度范围？岩性大类和岩性小类分别是什么？ | 全新世有哪些钻孔层位记录？请列出对应分层信息。 | geologic_epoch | 全新世 | stratigraphy_exact_retrieval | 257 |
| qa_008 | 钻孔 CHGC008 的分层 CHGC008_3 位于哪个深度范围？岩性大类和岩性小类分别是什么？ | 给出更新世对应的所有钻孔、层位和分层信息。 | geologic_epoch | 更新世 | stratigraphy_exact_retrieval | 268 |
| qa_009 | 钻孔 CHGC009 的分层 CHGC009_2 位于哪个深度范围？岩性大类和岩性小类分别是什么？ | 列出第四纪对应的钻孔及分层信息。 | geologic_period | 第四纪 | stratigraphy_exact_retrieval | 525 |
| qa_010 | 钻孔 CHGC010 的分层 CHGC010_2 位于哪个深度范围？岩性大类和岩性小类分别是什么？ | 给出第四纪对应的所有钻孔、层位和分层信息。 | geologic_period | 第四纪 | stratigraphy_exact_retrieval | 525 |

## v4 测评指标

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| Stratigraphy 路由命中率 | 10/10 | 是否成功进入新路由 |
| 检索召回率 | 82.86% | layer_id 级别 |
| 检索准确率 | 100.0% | layer_id 级别 |
| 答案准确率 | 0.0% | Markdown 表格字段完整性 |
| 零幻觉率 | 100.0% | 是否存在额外编造 |
| 原有 40 条回归数 | 0 | 对比 v3 |

## Badcase 明细

| case_id | question | badcase_type | expected | actual | suspected_reason | suggested_fix |
| --- | --- | --- | --- | --- | --- | --- |
| qa_001 | 列出晚更新世冲洪积层对应的钻孔及分层信息。 | answer_format_error | route=stratigraphy_exact_retrieval, layers=6 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=6, recall=1.0, precision=1.0 | 检索正确但 Markdown 表格没有完整呈现纪、世、组、段、厚度等字段。 | 扩展 build_stratigraphy_direct_answer() 表格列，显式展示层位 ID、深度、厚度、纪、世、组、段、岩性。 |
| qa_002 | 灯笼沙段有哪些钻孔？请给出对应层位和分层信息。 | answer_format_error | route=stratigraphy_exact_retrieval, layers=60 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=60, recall=1.0, precision=1.0 | 检索正确但 Markdown 表格没有完整呈现纪、世、组、段、厚度等字段。 | 扩展 build_stratigraphy_direct_answer() 表格列，显式展示层位 ID、深度、厚度、纪、世、组、段、岩性。 |
| qa_003 | 给出人工填土在各钻孔中的分布情况和层位信息。 | answer_format_error | route=stratigraphy_exact_retrieval, layers=53 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=53, recall=1.0, precision=1.0 | 检索正确但 Markdown 表格没有完整呈现纪、世、组、段、厚度等字段。 | 扩展 build_stratigraphy_direct_answer() 表格列，显式展示层位 ID、深度、厚度、纪、世、组、段、岩性。 |
| qa_004 | 列出光明村层相关的所有钻孔、层位和分层信息。 | answer_format_error | route=stratigraphy_exact_retrieval, layers=10 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=10, recall=1.0, precision=1.0 | 检索正确但 Markdown 表格没有完整呈现纪、世、组、段、厚度等字段。 | 扩展 build_stratigraphy_direct_answer() 表格列，显式展示层位 ID、深度、厚度、纪、世、组、段、岩性。 |
| qa_005 | 给出晚更新世冲洪积层在各钻孔中的分布情况和层位信息。 | answer_format_error | route=stratigraphy_exact_retrieval, layers=6 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=6, recall=1.0, precision=1.0 | 检索正确但 Markdown 表格没有完整呈现纪、世、组、段、厚度等字段。 | 扩展 build_stratigraphy_direct_answer() 表格列，显式展示层位 ID、深度、厚度、纪、世、组、段、岩性。 |
| qa_006 | 列出人工填土对应的钻孔及分层信息。 | answer_format_error | route=stratigraphy_exact_retrieval, layers=53 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=53, recall=1.0, precision=1.0 | 检索正确但 Markdown 表格没有完整呈现纪、世、组、段、厚度等字段。 | 扩展 build_stratigraphy_direct_answer() 表格列，显式展示层位 ID、深度、厚度、纪、世、组、段、岩性。 |
| qa_007 | 全新世有哪些钻孔层位记录？请列出对应分层信息。 | retrieval_partial | route=stratigraphy_exact_retrieval, layers=257 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=200, recall=0.7782, precision=1.0 | Cypher 条件、LIMIT 或词表匹配导致部分 layer_id 漏召回。 | 检查 retrieve_stratigraphy_exact() 的 LIMIT、排序和多字段过滤条件。 |
| qa_007 | 全新世有哪些钻孔层位记录？请列出对应分层信息。 | answer_format_error | route=stratigraphy_exact_retrieval, layers=257 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=200, recall=0.7782, precision=1.0 | 检索正确但 Markdown 表格没有完整呈现纪、世、组、段、厚度等字段。 | 扩展 build_stratigraphy_direct_answer() 表格列，显式展示层位 ID、深度、厚度、纪、世、组、段、岩性。 |
| qa_008 | 给出更新世对应的所有钻孔、层位和分层信息。 | retrieval_partial | route=stratigraphy_exact_retrieval, layers=268 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=200, recall=0.7463, precision=1.0 | Cypher 条件、LIMIT 或词表匹配导致部分 layer_id 漏召回。 | 检查 retrieve_stratigraphy_exact() 的 LIMIT、排序和多字段过滤条件。 |
| qa_008 | 给出更新世对应的所有钻孔、层位和分层信息。 | answer_format_error | route=stratigraphy_exact_retrieval, layers=268 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=200, recall=0.7463, precision=1.0 | 检索正确但 Markdown 表格没有完整呈现纪、世、组、段、厚度等字段。 | 扩展 build_stratigraphy_direct_answer() 表格列，显式展示层位 ID、深度、厚度、纪、世、组、段、岩性。 |
| qa_009 | 列出第四纪对应的钻孔及分层信息。 | retrieval_partial | route=stratigraphy_exact_retrieval, layers=525 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=200, recall=0.381, precision=1.0 | Cypher 条件、LIMIT 或词表匹配导致部分 layer_id 漏召回。 | 检查 retrieve_stratigraphy_exact() 的 LIMIT、排序和多字段过滤条件。 |
| qa_009 | 列出第四纪对应的钻孔及分层信息。 | answer_format_error | route=stratigraphy_exact_retrieval, layers=525 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=200, recall=0.381, precision=1.0 | 检索正确但 Markdown 表格没有完整呈现纪、世、组、段、厚度等字段。 | 扩展 build_stratigraphy_direct_answer() 表格列，显式展示层位 ID、深度、厚度、纪、世、组、段、岩性。 |
| qa_010 | 给出第四纪对应的所有钻孔、层位和分层信息。 | retrieval_partial | route=stratigraphy_exact_retrieval, layers=525 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=200, recall=0.381, precision=1.0 | Cypher 条件、LIMIT 或词表匹配导致部分 layer_id 漏召回。 | 检查 retrieve_stratigraphy_exact() 的 LIMIT、排序和多字段过滤条件。 |
| qa_010 | 给出第四纪对应的所有钻孔、层位和分层信息。 | answer_format_error | route=stratigraphy_exact_retrieval, layers=525 | route=stratigraphy_exact_retrieval, rerank=stratigraphy_rule, layers=200, recall=0.381, precision=1.0 | 检索正确但 Markdown 表格没有完整呈现纪、世、组、段、厚度等字段。 | 扩展 build_stratigraphy_direct_answer() 表格列，显式展示层位 ID、深度、厚度、纪、世、组、段、岩性。 |

## 修复建议

1. P0：优先修复 route_miss、retrieval_empty、retrieval_noise 和 route_false_positive。
2. P1：修复 answer_format_error，保证直接答案表格完整展示纪、世、组、段、厚度和岩性字段。
3. P2：补充 Stratigraphy 专项单测，覆盖词表加载、Cypher 查询和 Markdown 输出格式。

## 可复现命令

```bash
python evals/scripts/build_stratigraphy_eval_v4.py
python evals/eval_retrieval_v2.py --dataset evals/datasets/rag_mvp_eval_50_stratigraphy_v4.jsonl --output-dir evals/results/v4_stratigraphy --rankgpt off
python evals/scripts/analyze_stratigraphy_v4.py --dataset evals/datasets/rag_mvp_eval_50_stratigraphy_v4.jsonl --eval-details evals/results/v4_stratigraphy/retrieval_v2_rankgpt_off_details.jsonl --output-dir evals/results/v4_stratigraphy
```
