from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from spy_game.config import RuntimeContext
from spy_game.engine_nodes import (
    node_assign_roles,
    node_begin_speech_round,
    node_finalize_speeches,
    node_generate_words,
    node_judge_result,
    node_show_final_result,
    node_speech_turn,
    node_vote_begin,
    node_vote_order_preview,
    node_vote_turn,
    route_after_begin,
    route_after_judge,
    route_after_speech,
    route_after_vote,
)
from spy_game.state_types import GameState


def _dummy_runtime() -> RuntimeContext:
    from unittest.mock import MagicMock

    from langchain_core.output_parsers import StrOutputParser

    from spy_game.config import GameConfig, RuntimeContext

    return RuntimeContext(
        llm=MagicMock(),
        parser=StrOutputParser(),
        game=GameConfig(num_players=4, num_undercover=1),
    )


def _bind(ctx: RuntimeContext, fn: Callable[[GameState, RuntimeContext], dict]) -> Callable[[GameState], dict]:
    def inner(state: GameState) -> dict:
        return fn(state, ctx)

    return inner


def build_state_graph(ctx: RuntimeContext) -> StateGraph:
    graph = StateGraph(GameState)
    graph.add_node("generate_words", _bind(ctx, node_generate_words))
    graph.add_node("assign_roles", _bind(ctx, node_assign_roles))
    graph.add_node("begin_speech_round", _bind(ctx, node_begin_speech_round))
    graph.add_node("speech_turn", _bind(ctx, node_speech_turn))
    graph.add_node("finalize_speeches", _bind(ctx, node_finalize_speeches))
    graph.add_node("vote_begin", _bind(ctx, node_vote_begin))
    graph.add_node("vote_order_preview", _bind(ctx, node_vote_order_preview))
    graph.add_node("vote_turn", _bind(ctx, node_vote_turn))
    graph.add_node("judge_result", _bind(ctx, node_judge_result))
    graph.add_node("show_final_result", _bind(ctx, node_show_final_result))

    graph.add_edge(START, "generate_words")
    graph.add_edge("generate_words", "assign_roles")
    graph.add_edge("assign_roles", "begin_speech_round")
    graph.add_conditional_edges("begin_speech_round", route_after_begin)
    graph.add_conditional_edges("speech_turn", route_after_speech)
    graph.add_edge("finalize_speeches", "vote_begin")
    graph.add_edge("vote_begin", "vote_order_preview")
    graph.add_edge("vote_order_preview", "vote_turn")
    graph.add_conditional_edges("vote_turn", route_after_vote)
    graph.add_conditional_edges("judge_result", route_after_judge)
    graph.add_edge("show_final_result", END)
    return graph


def compile_game_graph(ctx: RuntimeContext, checkpointer: Any | None = None) -> Any:
    return build_state_graph(ctx).compile(checkpointer=checkpointer)


def get_graph_mermaid() -> str:
    """静态拓扑 Mermaid 文本（不发起真实模型调用）。"""
    return build_state_graph(_dummy_runtime()).compile().get_graph().draw_mermaid()


def export_graph_topology_png(out_dir: Path, filename: str = "langgraph_topology.png") -> Path:
    """
    将 LangGraph 拓扑导出为 PNG（get_graph().draw_mermaid_png()），并写入 out_dir。
    图片可与 Streamlit `st.image` 直接加载；默认路径示例：项目根下 `.cache/langgraph_topology.png`。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    png = build_state_graph(_dummy_runtime()).compile().get_graph().draw_mermaid_png()
    out_path.write_bytes(png)
    return out_path
