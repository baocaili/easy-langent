from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
from dotenv import load_dotenv

from spy_game.config import GameConfig, LLMConfig, RuntimeContext
from spy_game.graph_build import export_graph_topology_png
from spy_game.llm_factory import load_llm
from spy_game.runner import (
    build_app,
    extract_interrupt_payload,
    get_snapshot,
    is_game_finished,
    resume_game,
    start_new_game,
)

st.set_page_config(page_title="谁是卧底 · LangGraph", layout="wide")
_root = Path(__file__).resolve().parent
load_dotenv(_root / ".env")
load_dotenv(_root.parent / ".env")


def _default_llm_cfg() -> LLMConfig:
    return LLMConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_API_BASE", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "800")),
    )


def _sidebar_llm() -> LLMConfig:
    st.subheader("大模型配置")
    d = _default_llm_cfg()
    api_key = st.text_input("OPENAI_API_KEY", value=d.api_key, type="password")
    base_url = st.text_input("OPENAI_API_BASE（可留空）", value=d.base_url)
    model = st.text_input("OPENAI_MODEL", value=d.model or "gpt-4o-mini")
    temperature = st.slider("temperature", 0.0, 1.5, float(d.temperature), 0.05)
    max_tokens = st.number_input("max_tokens", 200, 32000, int(d.max_tokens), 50)
    return LLMConfig(
        api_key=api_key.strip(),
        base_url=base_url.strip(),
        model=model.strip(),
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )


def _sidebar_game() -> GameConfig:
    st.subheader("人数与角色")
    n = st.slider("参与人数", 4, 10, 5)
    if n <= 6:
        st.caption("人数 ≤ 6：固定 1 名卧底。")
        uc = 1
    else:
        uc = st.radio("卧底人数（仅当人数 > 6）", options=[1, 2], index=0, horizontal=True)
    human = st.selectbox(
        "人类玩家（观战则不选）",
        options=["（无，全 AI 观战）"] + [f"P{i}" for i in range(1, n + 1)],
        index=0,
    )
    human_id = None if human.startswith("（") else human
    max_rounds = st.number_input("最大轮次（防死循环，超时判平民胜）", 5, 80, 30)
    return GameConfig(
        num_players=n,
        num_undercover=int(uc),
        human_player_id=human_id,
        max_rounds=int(max_rounds),
    )


@st.cache_resource
def _cached_graph_png_path(project_root_str: str) -> str:
    """首次生成 PNG 写入 `.cache/`，进程内复用路径。"""
    return str(export_graph_topology_png(Path(project_root_str) / ".cache"))


def _round_speech_lines(
    state: Dict[str, Any], speech_order: Optional[list[str]] = None
) -> list[tuple[str, str]]:
    """按发言顺序返回 (玩家ID, 发言) 列表；顺序未知时补全剩余玩家。"""
    speeches = dict(state.get("speeches") or {})
    order = list(speech_order if speech_order is not None else state.get("speech_order") or [])
    lines: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pid in order:
        if pid in speeches:
            lines.append((pid, speeches[pid]))
            seen.add(pid)
    for pid in sorted(speeches.keys()):
        if pid not in seen:
            lines.append((pid, speeches[pid]))
    return lines


def _render_current_round_speeches(state: Dict[str, Any], intr: Optional[Dict[str, Any]] = None) -> None:
    order = intr.get("speech_order") if intr else None
    lines = _round_speech_lines(state, order)
    if not lines:
        st.caption("（本轮发言尚未出现在状态中）")
        return
    st.markdown("##### 本轮发言实录")
    for pid, text in lines:
        st.markdown(f"**{pid}**")
        st.write(text)


def main() -> None:
    if "ctx" not in st.session_state:
        st.session_state.ctx = None
    if "app" not in st.session_state:
        st.session_state.app = None
    if "config" not in st.session_state:
        st.session_state.config = None
    if "pending_payload" not in st.session_state:
        st.session_state.pending_payload = None

    st.title("谁是卧底 · LangGraph + Streamlit")
    st.caption("支持全 AI 观战、人类入局、checkpoint + interrupt 恢复。")
    start_status = st.empty()

    with st.sidebar:
        llm_cfg = _sidebar_llm()
        game_cfg = _sidebar_game()
        if st.button("测试模型连接", width="stretch"):
            try:
                llm, _ = load_llm(llm_cfg)
                r = llm.invoke("严格回复分号之后的内容：大模型测试成功！")
                st.success(getattr(r, "content", str(r))[:200])
            except Exception as e:
                st.error(str(e))

        start = st.button(
            "开始新对局",
            type="primary",
            width="stretch",
            key="btn_new_game",
        )

    tab_play, tab_graph, tab_log = st.tabs(["对局", "LangGraph 可视化", "日志与状态"])

    with tab_graph:
        st.markdown("### LangGraph 拓扑图（PNG）")
        st.caption("由 `get_graph().draw_mermaid_png()` 生成，缓存在项目目录 `.cache/langgraph_topology.png`。")
        try:
            png_path = _cached_graph_png_path(str(_root))
            st.image(png_path, width="stretch")
        except Exception as e:
            st.warning(f"PNG 生成或加载失败：{e}。可点击「刷新拓扑图」重试。")
        if st.button("刷新拓扑图（改代码后）", key="btn_refresh_graph_png"):
            _cached_graph_png_path.clear()
            st.rerun()
        st.markdown("### 本局节点执行顺序（动态）")
        st.caption("每步节点会向 `RuntimeContext.trace` 追加记录。")
        if st.session_state.get("ctx") and getattr(st.session_state.ctx, "trace", None):
            st.code(" → ".join(st.session_state.ctx.trace))
        else:
            st.caption("开局后将显示节点执行轨迹。")

    if start:
        start_status.info("正在开局：会多次调用大模型，全 AI 一局可能要 **1～5 分钟**，请勿关闭页面…")
        try:
            with st.spinner("对局进行中，请稍候…"):
                llm, parser = load_llm(llm_cfg)
                ctx = RuntimeContext(llm=llm, parser=parser, game=game_cfg)
                app = build_app(ctx)
                players = game_cfg.player_ids()
                state, config = start_new_game(app, ctx, players)
            st.session_state.ctx = ctx
            st.session_state.app = app
            st.session_state.config = config
            st.session_state.pending_payload = None
            snap = get_snapshot(app, config)
            st.session_state.pending_payload = extract_interrupt_payload(snap)
            st.session_state.last_state = state
            st.session_state.elim_seen_len = len(state.get("eliminated") or [])
            start_status.success("开局完成。已切换到对局数据，下方可查看「对局」「日志与状态」。")
            st.rerun()
        except Exception as e:
            start_status.error(f"开局失败：{e}")

    ctx: Optional[RuntimeContext] = st.session_state.ctx
    app: Any = st.session_state.app
    config: Optional[Dict[str, Any]] = st.session_state.config

    with tab_log:
        if ctx:
            st.subheader("运行日志")
            st.text_area("log", value="\n".join(ctx.log), height=240)
            st.subheader("节点 trace")
            st.write(ctx.trace or [])
            if config and app:
                snap = get_snapshot(app, config)
                st.subheader("Checkpoint 状态快照（values）")
                st.json(dict(snap.values) if snap and snap.values else {})
        else:
            st.info("尚未开始游戏。")

    with tab_play:
        if not ctx or not app or not config:
            st.info(
                "在左侧配置后点击「开始新对局」。开局后页面会刷新；若长时间无反应，请看页顶提示或「日志与状态」。"
                " 默认本机地址一般为 `http://localhost:8501`。"
            )
        else:
            _render_tab_play(ctx, app, config)


def _render_tab_play(ctx: RuntimeContext, app: Any, config: Dict[str, Any]) -> None:
    snap = get_snapshot(app, config)
    state = dict(snap.values) if snap and snap.values else {}
    st.session_state.last_state = state

    intr = extract_interrupt_payload(snap)
    if intr is None and snap:
        intr = st.session_state.pending_payload
    else:
        st.session_state.pending_payload = intr

    human_id = ctx.game.human_player_id
    eliminated = list(state.get("eliminated") or [])
    seen_elim = int(st.session_state.get("elim_seen_len", 0))
    if human_id and len(eliminated) > seen_elim:
        pid = eliminated[seen_elim]
        ra = state.get("role_assignment") or {}
        role_t = ra.get(pid)
        role = role_t[0] if role_t else "未知"
        st.subheader("投票结果 · 本轮淘汰")
        if pid == human_id:
            st.error("很遗憾，你被淘汰了！继续努力！")
            st.info(f"你的角色为：**{role}**。")
        else:
            st.warning(f"本轮投票被淘汰玩家：**{pid}**")
            st.info(f"该玩家角色为：**{role}**。")
        if st.button("知道了", key=f"btn_ack_elim_{seen_elim}", width="stretch"):
            st.session_state.elim_seen_len = len(eliminated)
            st.rerun()
        st.divider()
        return

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.metric("轮次", state.get("round", 1))
        st.metric("状态", state.get("game_status", ""))
    with col_b:
        if is_game_finished(state):
            st.success(f"本局结束，胜利方：{'平民' if state.get('winner')=='civilian' else '卧底'}")
            st.write("平民词：", state.get("civilian_word"))
            st.write("卧底词：", state.get("undercover_word"))
            st.write("淘汰顺序：", state.get("eliminated"))

    st.markdown("---")
    if is_game_finished(state):
        st.success("对局已结束。可在侧栏点击「开始新对局」。")
        return

    if intr and intr.get("phase") == "recap":
        st.subheader(f"第 {intr.get('round', 1)} 轮 · 发言已结束")
        _render_current_round_speeches(state, intr)
        st.caption(
            "请阅读全员发言后点击下方按钮。此步**不产生模型调用**；下一步会先展示随机抽签的投票顺序。"
            "随后每位 AI 投票都会调用一次大模型——若您的顺位较靠后，等待时间会叠加。"
        )
        if st.button("确认完毕，进入投票", key="btn_recap", width="stretch"):
            try:
                with st.spinner("正在生成本轮投票顺序…"):
                    resume_game(app, config, True, ctx)
                snap2 = get_snapshot(app, config)
                st.session_state.pending_payload = extract_interrupt_payload(snap2)
                st.rerun()
            except Exception as e:
                st.error(str(e))
    elif intr and intr.get("phase") == "vote_order":
        rnd = intr.get("round", 1)
        order = list(intr.get("order") or [])
        hi = int(intr.get("human_index", -1))
        n_ai = int(intr.get("ai_votes_before_human") or 0)
        st.subheader(f"第 {rnd} 轮 · 投票顺序已生成")
        st.code(" → ".join(order) if order else "（空）")
        if hi >= 0 and order:
            st.caption(f"您在顺序中为第 **{hi + 1}** / {len(order)} 位投票。")
        if n_ai > 0:
            st.warning(
                f"在您轮到之前，**{n_ai}** 位 AI 将依次投票（每位 **1 次**大模型调用），通常需要一至数分钟。"
            )
        else:
            st.info("您最先投票；点击下方按钮后将直接进入您的投票界面。")
        if st.button("开始投票", key="btn_vote_order", width="stretch"):
            try:
                spin_msg = (
                    f"AI 依次投票中（约 {n_ai} 次模型调用），请稍候…"
                    if n_ai > 0
                    else "正在进入投票…"
                )
                with st.spinner(spin_msg):
                    resume_game(app, config, True, ctx)
                snap2 = get_snapshot(app, config)
                st.session_state.pending_payload = extract_interrupt_payload(snap2)
                st.rerun()
            except Exception as e:
                st.error(str(e))
    elif intr and intr.get("phase") == "speech":
        st.subheader(f"轮到你的发言：{intr.get('player')}")
        st.write(f"你的角色：**{intr.get('role')}**，词语：**{intr.get('word')}**")
        txt = st.text_area("发言内容", key="human_speech")
        if st.button("提交发言", key="btn_speech"):
            try:
                with st.spinner("其余玩家（AI）正在依次发言，请稍候（通常需一至数分钟）…"):
                    resume_game(app, config, txt.strip(), ctx)
                snap2 = get_snapshot(app, config)
                st.session_state.pending_payload = extract_interrupt_payload(snap2)
                st.rerun()
            except Exception as e:
                st.error(str(e))
    elif intr and intr.get("phase") == "vote":
        st.subheader(f"轮到你的投票：{intr.get('voter')}")
        _render_current_round_speeches(state, None)
        st.divider()
        targets = intr.get("targets") or []
        choice = st.selectbox("投给谁", options=targets, key="human_vote")
        if st.button("提交投票", key="btn_vote"):
            try:
                with st.spinner("正在处理投票（若之后仍有 AI 投票，请稍候）…"):
                    resume_game(app, config, str(choice), ctx)
                snap2 = get_snapshot(app, config)
                st.session_state.pending_payload = extract_interrupt_payload(snap2)
                st.rerun()
            except Exception as e:
                st.error(str(e))
    else:
        st.info("对局进行中（全 AI 步进）。若长时间无输出，请查看「日志与状态」页。")
        if st.button("尝试自动步进（无人类时应已结束）"):
            st.rerun()


if __name__ == "__main__":
    main()
