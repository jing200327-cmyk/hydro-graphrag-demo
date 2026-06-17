# app.py
# ============================================================
# Streamlit 前端：面向岩土工程与水文地质勘查资料的 GraphRAG 智能问答系统
# ============================================================

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from qa_backend import run_qa
from src.conversation.service import ConversationService


# ============================================================
# 1. 页面基础配置
# ============================================================

st.set_page_config(
    page_title="面向岩土工程与水文地质勘查资料的 GraphRAG 智能问答系统",
    page_icon="💧",
    layout="wide",
)


# ============================================================
# 2. 页面样式
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .sub-title {
        color: #666666;
        font-size: 15px;
        margin-bottom: 18px;
    }

    .hint-box {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px 16px;
        background-color: #fafafa;
        margin-bottom: 14px;
    }

    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px;
        background-color: #ffffff;
    }

    .small-note {
        color: #777777;
        font-size: 13px;
    }

    .chunk-title {
        font-weight: 700;
        font-size: 15px;
    }

    .chunk-meta {
        color: #777777;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. Session State 初始化
# ============================================================

def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    if "run_count" not in st.session_state:
        st.session_state.run_count = 0

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = ""

    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = set()


init_session_state()

conversation_service = ConversationService()

if not st.session_state.conversation_id:
    st.session_state.conversation_id = conversation_service.create_conversation(
        title="新会话",
        project_name="hydro_graphrag_streamlit",
    )

if not conversation_service.conversation_exists(st.session_state.conversation_id):
    st.session_state.conversation_id = conversation_service.create_conversation(
        title="新会话",
        project_name="hydro_graphrag_streamlit",
    )

st.session_state.run_count = len(
    conversation_service.load_messages(st.session_state.conversation_id)
) // 2


# ============================================================
# 4. 工具函数
# ============================================================

def safe_dataframe(data: Any) -> pd.DataFrame:
    """
    将 list[dict]、dict 或其他结构安全转为 DataFrame。
    """

    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, list):
        if len(data) == 0:
            return pd.DataFrame()
        return pd.DataFrame(data)

    if isinstance(data, dict):
        return pd.DataFrame([data])

    return pd.DataFrame([{"value": str(data)}])


def to_json_str(data: Any) -> str:
    """
    将对象转为中文友好的 JSON 字符串。
    """

    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(data)


def render_json_download(result: Dict[str, Any], filename_prefix: str = "qa_result") -> None:
    """
    下载完整 JSON 结果。
    """

    json_str = to_json_str(result)
    filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    st.download_button(
        label="📥 下载完整 JSON 结果",
        data=json_str,
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


def render_markdown_download(text: str, filename_prefix: str = "final_answer") -> None:
    """
    下载 Markdown 文本。
    """

    filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    st.download_button(
        label="📥 下载最终回答 Markdown",
        data=text or "",
        file_name=filename,
        mime="text/markdown",
        use_container_width=True,
    )


def render_turn_feedback(turn_id: str, key_prefix: str = "") -> None:
    turn_id = str(turn_id or "").strip()

    if not turn_id:
        return

    submitted_key = f"{key_prefix}:{turn_id}"

    if submitted_key in st.session_state.feedback_submitted:
        st.caption("已记录本轮反馈。")
        return

    st.caption("本轮回答反馈")
    tag = st.selectbox(
        "问题归因标签",
        options=[
            "检索不准",
            "图谱缺数据",
            "回答不完整",
            "兜底错误",
            "事实字段错误",
        ],
        key=f"feedback_tag_{submitted_key}",
    )
    comment = st.text_input(
        "补充说明（可选）",
        key=f"feedback_comment_{submitted_key}",
    )
    col_helpful, col_bad = st.columns(2)

    with col_helpful:
        if st.button("👍 有帮助", key=f"feedback_helpful_{submitted_key}", use_container_width=True):
            conversation_service.add_feedback(
                turn_id=turn_id,
                rating=5,
                tag="有帮助",
                comment=comment,
                metadata={"ui": "streamlit"},
            )
            st.session_state.feedback_submitted.add(submitted_key)
            st.rerun()

    with col_bad:
        if st.button("👎 不准确", key=f"feedback_bad_{submitted_key}", use_container_width=True):
            conversation_service.add_feedback(
                turn_id=turn_id,
                rating=-1,
                tag=tag,
                comment=comment,
                metadata={"ui": "streamlit"},
            )
            st.session_state.feedback_submitted.add(submitted_key)
            st.rerun()


def get_score_value(item: Dict[str, Any]) -> Any:
    """
    兼容不同字段名的分数展示。
    """

    for key in ["score", "vector_score", "final_score", "semantic_score"]:
        if key in item:
            return item.get(key)
    return ""


# ============================================================
# 5. 各结果模块渲染函数
# ============================================================

def render_question_analysis(question_analysis: Dict[str, Any]) -> None:
    st.subheader("📌 问题解析结果")

    if not question_analysis:
        st.info("暂无问题解析结果。")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**核心实体**")
        st.write(
            question_analysis.get("核心实体")
            or question_analysis.get("core_entities")
            or []
        )

        st.markdown("**问题意图**")
        st.write(
            question_analysis.get("问题意图")
            or question_analysis.get("intent")
            or ""
        )

    with col2:
        st.markdown("**检索关键词**")
        st.write(
            question_analysis.get("检索关键词")
            or question_analysis.get("keywords")
            or []
        )

        st.markdown("**需要关注的水文地质因素**")
        st.write(
            question_analysis.get("需要关注的水文地质因素")
            or question_analysis.get("hydro_factors")
            or []
        )

    st.markdown("**建议检索方向**")
    st.write(
        question_analysis.get("建议检索方向")
        or question_analysis.get("search_direction")
        or ""
    )

    with st.expander("查看问题解析原始 JSON", expanded=False):
        st.json(question_analysis)


def render_evidence_chunks(chunks: List[Dict[str, Any]]) -> None:
    st.subheader("📚 EvidenceChunk 向量召回结果")

    if not chunks:
        st.info("暂无 EvidenceChunk 召回结果。")
        return

    df = safe_dataframe(chunks)

    show_columns = [
        col for col in [
            "chunk_id",
            "chunk_type",
            "source_label",
            "source_id",
            "vector_score",
            "score",
            "chunk_text",
            "graph_context_text",
        ]
        if col in df.columns
    ]

    if show_columns:
        st.dataframe(df[show_columns], use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    st.markdown("### 证据片段详情")

    for idx, item in enumerate(chunks, start=1):
        chunk_id = item.get("chunk_id", f"chunk_{idx}")
        chunk_type = item.get("chunk_type", "")
        source_label = item.get("source_label", "")
        source_id = item.get("source_id", "")
        score_value = get_score_value(item)

        title_parts = [f"{idx}. {chunk_id}"]

        if chunk_type:
            title_parts.append(str(chunk_type))

        if score_value != "":
            title_parts.append(f"score={score_value}")

        title = " | ".join(title_parts)

        with st.expander(title, expanded=(idx <= 2)):
            st.markdown("**EvidenceChunk 文本**")
            st.write(item.get("chunk_text", ""))

            meta_col1, meta_col2, meta_col3 = st.columns(3)

            with meta_col1:
                st.markdown("**chunk_type**")
                st.write(chunk_type)

            with meta_col2:
                st.markdown("**source_label**")
                st.write(source_label)

            with meta_col3:
                st.markdown("**source_id**")
                st.write(source_id)

            if item.get("graph_context_text"):
                st.markdown("**图谱扩展文本**")
                st.write(item.get("graph_context_text"))

            with st.expander("查看该 chunk 原始 JSON", expanded=False):
                st.json(item)


def render_graph_context(graph_context: List[Dict[str, Any]], retrieved_chunks: List[Dict[str, Any]]) -> None:
    st.subheader("🕸 图谱扩展结果")

    if not graph_context and not retrieved_chunks:
        st.info("暂无图谱扩展结果。")
        return

    if graph_context:
        df = safe_dataframe(graph_context)

        show_columns = [
            col for col in [
                "chunk_id",
                "graph_context_text",
                "source",
                "relation",
                "target",
            ]
            if col in df.columns
        ]

        if show_columns:
            st.dataframe(df[show_columns], use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

    st.markdown("### 按 EvidenceChunk 展示图谱上下文")

    has_any_context = False

    for idx, item in enumerate(retrieved_chunks or [], start=1):
        chunk_id = item.get("chunk_id", f"chunk_{idx}")
        graph_context = item.get("graph_context")
        graph_context_text = item.get("graph_context_text")

        if not graph_context and not graph_context_text:
            continue

        has_any_context = True

        with st.expander(f"{chunk_id} 的图谱扩展上下文", expanded=(idx <= 2)):
            if graph_context_text:
                st.markdown("**图谱扩展文本**")
                st.write(graph_context_text)

            if graph_context:
                st.markdown("**图谱扩展 JSON**")
                st.json(graph_context)

    if not has_any_context:
        st.info("当前召回结果中未包含 graph_context 或 graph_context_text。")


def render_b_score_results(b_results: List[Dict[str, Any]]) -> None:
    st.subheader("📊 B 综合相关性评分")

    if not b_results:
        st.info("暂无 B 评分结果。")
        return

    df = safe_dataframe(b_results)
    st.dataframe(df, use_container_width=True)

    score_col = None
    for candidate in ["score", "b_score", "total_score"]:
        if candidate in df.columns:
            score_col = candidate
            break

    if "chunk_id" in df.columns and score_col:
        chart_df = df[["chunk_id", score_col]].copy()
        chart_df = chart_df.set_index("chunk_id")
        st.bar_chart(chart_df)

    st.markdown("### B 评分解释")

    for idx, item in enumerate(b_results, start=1):
        chunk_id = item.get("chunk_id", f"chunk_{idx}")
        score = (
            item.get("score")
            or item.get("b_score")
            or item.get("total_score")
            or ""
        )

        with st.expander(f"{chunk_id} | B score={score}", expanded=(idx <= 2)):
            st.markdown("**命中实体**")
            st.write(item.get("matched_entities", []))

            st.markdown("**命中渗透率因素**")
            st.write(item.get("matched_permeability_factors", []))

            st.markdown("**是否保留**")
            st.write(item.get("keep", ""))

            st.markdown("**评分原因**")
            st.write(item.get("reason", ""))

            with st.expander("查看该 B 评分原始 JSON", expanded=False):
                st.json(item)


def render_c_rerank_results(c_results: List[Dict[str, Any]]) -> None:
    st.subheader("🧠 C 水文地质规则重排序")

    if not c_results:
        st.info("暂无 C 重排序结果。")
        return

    df = safe_dataframe(c_results)
    st.dataframe(df, use_container_width=True)

    score_col = None
    for candidate in ["final_score", "c_score", "score"]:
        if candidate in df.columns:
            score_col = candidate
            break

    if "chunk_id" in df.columns and score_col:
        chart_df = df[["chunk_id", score_col]].copy()
        chart_df = chart_df.set_index("chunk_id")
        st.bar_chart(chart_df)

    st.markdown("### 水文规则解释")

    for idx, item in enumerate(c_results, start=1):
        chunk_id = item.get("chunk_id", f"chunk_{idx}")
        final_score = (
            item.get("final_score")
            or item.get("c_score")
            or item.get("score")
            or ""
        )

        with st.expander(f"{chunk_id} | C final_score={final_score}", expanded=(idx <= 2)):
            st.markdown("**语义相关性得分**")
            st.write(item.get("semantic_score", ""))

            st.markdown("**规则匹配得分**")
            st.write(item.get("rule_score", ""))

            st.markdown("**证据完整性得分**")
            st.write(item.get("evidence_score", ""))

            st.markdown("**命中的水文地质规则**")
            st.write(item.get("matched_rules", []))

            st.markdown("**冲突或不确定性**")
            st.write(item.get("conflict_or_uncertainty", ""))

            st.markdown("**重排序理由**")
            st.write(item.get("rerank_reason", ""))

            st.markdown("**是否用于最终上下文**")
            st.write(item.get("use_for_context", ""))

            with st.expander("查看该 C 重排序原始 JSON", expanded=False):
                st.json(item)


def render_final_prompt(final_prompt: str) -> None:
    st.subheader("📝 最终 Prompt")

    if not final_prompt:
        st.info("暂无 final_prompt。")
        return

    st.text_area(
        label="发送给 DeepSeek 的最终 Prompt",
        value=final_prompt,
        height=500,
    )

    st.download_button(
        label="📥 下载 final_prompt.txt",
        data=final_prompt,
        file_name=f"final_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )


def render_saved_outputs() -> None:
    st.subheader("📁 后端保存文件")

    output_dir = Path(r"C:\Users\MECHREVO\ana_exp\RAG\pe\output")

    if not output_dir.exists():
        st.info(f"暂未发现输出目录：{output_dir}")
        return

    files = [
        "block3_end_to_end_result.json",
        "block3_final_prompt.txt",
        "block3_final_answer.md",
        "block3_vector_and_graph_results.xlsx",
        "block3_b_score_results.xlsx",
        "block3_c_rerank_results.xlsx",
    ]

    found_any = False

    for filename in files:
        file_path = output_dir / filename

        if not file_path.exists():
            continue

        found_any = True

        with open(file_path, "rb") as f:
            st.download_button(
                label=f"📥 下载 {filename}",
                data=f.read(),
                file_name=filename,
                use_container_width=True,
            )

    if not found_any:
        st.info("输出目录存在，但暂未发现板块三输出文件。")


def render_result_tabs(result: Dict[str, Any]) -> None:
    """
    展示完整可解释结果。
    """

    retrieved_chunks = result.get("retrieved_chunks", [])
    graph_context = result.get("graph_context", [])

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "📌 问题解析",
            "📚 EvidenceChunk",
            "🕸 图谱扩展",
            "📊 B 评分",
            "🧠 C 重排序",
            "📝 Final Prompt",
            "🧾 完整 JSON",
            "📁 输出文件",
        ]
    )

    with tab1:
        render_question_analysis(result.get("question_analysis", {}))

    with tab2:
        render_evidence_chunks(retrieved_chunks)

    with tab3:
        render_graph_context(graph_context, retrieved_chunks)

    with tab4:
        render_b_score_results(result.get("b_score_results", []))

    with tab5:
        render_c_rerank_results(result.get("c_rerank_results", []))

    with tab6:
        render_final_prompt(result.get("final_prompt", ""))

    with tab7:
        st.subheader("🧾 完整结果 JSON")
        st.json(result)
        render_json_download(result)

    with tab8:
        render_saved_outputs()


# ============================================================
# 6. 页面标题
# ============================================================

st.markdown(
    """
    <div class="main-title">💧 面向岩土工程与水文地质勘查资料的 GraphRAG 智能问答系统</div>
    <div class="sub-title">
    基于 Neo4j Vector Index、EvidenceChunk、图谱扩展、B 综合评分、C 水文规则重排序和 DeepSeek 的专业问答界面
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hint-box">
    <b>建议提问方式：</b>
    可以围绕岩性、钻孔编号、层位编号、深度范围、渗透率等级、k_value、粒径、胶结、裂隙、夹层等进行提问。
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. 侧边栏控制面板
# ============================================================

with st.sidebar:
    st.header("⚙️ 控制面板")

    st.markdown("### 会话管理")

    current_conversation = conversation_service.get_conversation(st.session_state.conversation_id)
    current_title = current_conversation.get("title") or "新会话"

    st.caption(f"当前会话 ID：{st.session_state.conversation_id}")
    edited_title = st.text_input(
        "当前会话标题",
        value=current_title,
        key=f"conversation_title_{st.session_state.conversation_id}",
    )

    if st.button("保存会话标题", use_container_width=True):
        conversation_service.update_conversation_title(
            st.session_state.conversation_id,
            edited_title,
        )
        st.rerun()

    recent_conversations = conversation_service.list_conversations(limit=20)
    conversation_labels = []
    conversation_id_by_label = {}

    for item in recent_conversations:
        title = item.get("title") or "新会话"
        turn_count = item.get("turn_count") or 0
        short_id = str(item.get("conversation_id", ""))[-6:]
        label = f"{title}（{turn_count}轮，{short_id}）"
        conversation_labels.append(label)
        conversation_id_by_label[label] = item.get("conversation_id")

    current_label = next(
        (
            label
            for label, conv_id in conversation_id_by_label.items()
            if conv_id == st.session_state.conversation_id
        ),
        conversation_labels[0] if conversation_labels else "",
    )

    if conversation_labels:
        selected_conversation_label = st.selectbox(
            "历史会话列表",
            options=conversation_labels,
            index=conversation_labels.index(current_label),
        )
        selected_conversation_id = conversation_id_by_label[selected_conversation_label]

        if selected_conversation_id != st.session_state.conversation_id:
            st.session_state.conversation_id = selected_conversation_id
            st.session_state.last_result = None
            st.session_state.last_question = ""
            st.session_state.run_count = len(
                conversation_service.load_messages(selected_conversation_id)
            ) // 2
            st.rerun()

    if st.button("➕ 新建会话", use_container_width=True):
        st.session_state.conversation_id = conversation_service.create_conversation(
            title="新会话",
            project_name="hydro_graphrag_streamlit",
        )
        st.session_state.last_result = None
        st.session_state.last_question = ""
        st.session_state.run_count = 0
        st.rerun()

    export_json = conversation_service.export_conversation_json(st.session_state.conversation_id)
    export_markdown = conversation_service.export_conversation_markdown(st.session_state.conversation_id)
    export_base_name = f"conversation_{st.session_state.conversation_id[-8:]}"

    st.download_button(
        "导出当前会话 JSON",
        data=export_json,
        file_name=f"{export_base_name}.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        "导出当前会话 Markdown",
        data=export_markdown,
        file_name=f"{export_base_name}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    confirm_delete = st.checkbox("确认删除当前会话")
    if st.button("删除当前会话", use_container_width=True, disabled=not confirm_delete):
        conversation_service.delete_conversation(st.session_state.conversation_id)
        st.session_state.conversation_id = conversation_service.create_conversation(
            title="新会话",
            project_name="hydro_graphrag_streamlit",
        )
        st.session_state.last_result = None
        st.session_state.last_question = ""
        st.session_state.run_count = 0
        st.rerun()

    st.divider()

    st.markdown("### 检索与生成参数")

    raw_top_k = st.slider(
        "向量检索 raw_top_k",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Neo4j Vector Index 初始召回的 EvidenceChunk 数量。",
    )

    final_top_k = st.slider(
        "最终上下文 final_top_k",
        min_value=3,
        max_value=20,
        value=10,
        step=1,
        help="进入 B/C 评分和最终 Prompt 的主要证据数量。",
    )

    b_keep_threshold = st.slider(
        "B 评分保留阈值",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=5.0,
    )

    c_context_threshold = st.slider(
        "C 重排序上下文阈值",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=5.0,
    )

    max_tokens = st.slider(
        "DeepSeek max_tokens",
        min_value=512,
        max_value=4096,
        value=2048,
        step=256,
    )

    st.divider()

    st.markdown("### 示例问题")

    example_questions = [
        "CHGC001号钻孔有哪些分层，各个分层深度和岩性如何？",
        "CHGC011_2 位于哪个深度范围？岩性大类和岩性小类分别是什么？",
        "CHGC002号钻孔有哪些分层有渗透率观测值？",
        "为什么水文地质特征“泥质胶结”会对渗透率产生“明显降低”的影响？请结合资料中的规则解释。",
        "钻孔 CHGC999 在 10m 至 12m 的岩性是什么",
        "哪些钻孔分层渗透率值较高",
        "给出横栏段的所有钻孔的位置及其分层信息",
    ]

    selected_example = None

    for question in example_questions:
        if st.button(question, use_container_width=True):
            selected_example = question

    st.divider()

    st.markdown("### 显示选项")

    show_explain = st.checkbox("显示可解释检索与推理结果", value=True)
    show_final_answer_download = st.checkbox("显示最终回答下载按钮", value=True)

    st.divider()

    if st.button("🧹 清空聊天记录", use_container_width=True):
        st.session_state.conversation_id = conversation_service.create_conversation(
            title="新会话",
            project_name="hydro_graphrag_streamlit",
        )
        st.session_state.last_result = None
        st.session_state.last_question = ""
        st.session_state.run_count = 0
        st.rerun()


# ============================================================
# 8. 展示历史聊天
# ============================================================

conversation_messages = conversation_service.load_messages(st.session_state.conversation_id)
st.session_state.run_count = len(conversation_messages) // 2

for msg in conversation_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_turn_feedback(
                turn_id=msg.get("turn_id", ""),
                key_prefix=f"history_{msg.get('turn_index')}",
            )


# ============================================================
# 9. 用户输入
# ============================================================

typed_question = st.chat_input("请输入地下水、岩性、渗透率、钻孔层位等相关问题...")

user_question = selected_example or typed_question


# ============================================================
# 10. 执行问答
# ============================================================

if user_question:
    st.session_state.last_question = user_question
    st.session_state.run_count += 1

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.status("正在执行 GraphRAG 问答流程...", expanded=True) as status:
            st.write("1. 解析用户问题")
            st.write("2. 对用户问题进行向量化")
            st.write(f"3. 在 Neo4j Vector Index 中召回 EvidenceChunk，raw_top_k={raw_top_k}")
            st.write("4. 对召回片段执行图谱扩展")
            st.write(f"5. 执行 B 综合相关性评分，阈值={b_keep_threshold}")
            st.write(f"6. 执行 C 水文地质规则重排序，阈值={c_context_threshold}")
            st.write("7. 构造最终 Prompt 并调用 DeepSeek")

            try:
                result = run_qa(
                    user_question=user_question,
                    top_k=final_top_k,
                    raw_top_k=raw_top_k,
                    b_keep_threshold=b_keep_threshold,
                    c_context_threshold=c_context_threshold,
                    max_tokens=max_tokens,
                    conversation_id=st.session_state.conversation_id,
                )

                status.update(
                    label="GraphRAG 问答完成",
                    state="complete",
                    expanded=False,
                )

            except Exception as e:
                status.update(
                    label="GraphRAG 问答失败",
                    state="error",
                    expanded=True,
                )

                st.error("后端调用失败。请查看 Streamlit 启动命令行中的报错信息。")
                st.exception(e)
                st.stop()

        final_answer = result.get("final_answer", "未生成最终答案。")

        if result.get("conversation_id"):
            st.session_state.conversation_id = result["conversation_id"]

        st.markdown(final_answer)

        if show_final_answer_download:
            render_markdown_download(final_answer)

        render_turn_feedback(
            turn_id=result.get("turn_id", ""),
            key_prefix="current",
        )

        st.session_state.last_result = result


# ============================================================
# 11. 展示可解释结果
# ============================================================

if st.session_state.last_result and show_explain:
    st.divider()
    st.header("🔍 可解释检索与推理结果")
    render_result_tabs(st.session_state.last_result)


# ============================================================
# 12. 底部运行信息
# ============================================================

with st.expander("运行信息", expanded=False):
    st.write("本次会话问题数：", st.session_state.run_count)
    st.write("最近一次问题：", st.session_state.last_question)
    st.write("前端状态：Streamlit 正常运行")
