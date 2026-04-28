from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from spy_game.config import GameConfig, RuntimeContext
from spy_game.graph_build import compile_game_graph
from spy_game.state_types import GameState, empty_state


def new_thread_id() -> str:
    return str(uuid.uuid4())


def build_app(ctx: RuntimeContext) -> Any:
    return compile_game_graph(ctx, checkpointer=MemorySaver())


def start_new_game(app: Any, ctx: RuntimeContext, players: list[str]) -> Tuple[GameState, Dict[str, Any]]:
    """新线程、从空状态开始执行，直到结束或遇到 interrupt。"""
    tid = new_thread_id()
    config: Dict[str, Any] = {"configurable": {"thread_id": tid}}
    init = empty_state(players)
    init["max_rounds"] = ctx.game.max_rounds
    ctx.trace.clear()
    ctx.log.clear()
    app.invoke(init, config)
    snap = app.get_state(config)
    return snap.values, config


def resume_game(app: Any, config: Dict[str, Any], value: Any, ctx: Optional[RuntimeContext] = None) -> GameState:
    if ctx is not None:
        ctx.trace.append("resume")
    app.invoke(Command(resume=value), config)
    snap = app.get_state(config)
    return snap.values


def get_snapshot(app: Any, config: Dict[str, Any]) -> Any:
    return app.get_state(config)


def _interrupt_items_from_field(ints: Any) -> list[Any]:
    if ints is None:
        return []
    if isinstance(ints, dict):
        out: list[Any] = []
        for v in ints.values():
            if isinstance(v, (list, tuple)):
                out.extend(v)
            else:
                out.append(v)
        return out
    if isinstance(ints, (list, tuple)):
        return list(ints)
    return [ints]


def extract_interrupt_payload(snap: Any) -> Optional[dict]:
    """从 checkpoint 快照解析 interrupt 负载（兼容 dict / tuple 等 LangGraph 版本）。"""
    if snap is None:
        return None
    try:
        vals = getattr(snap, "values", None) or {}
        if isinstance(vals, dict):
            raw = vals.get("__interrupt__")
            if raw:
                seq = raw if isinstance(raw, (list, tuple)) else [raw]
                for it in seq:
                    val = getattr(it, "value", None)
                    if val is None and isinstance(it, dict):
                        val = it
                    if isinstance(val, dict):
                        return val
                    if val is not None:
                        return {"phase": "raw", "value": val}

        ints = getattr(snap, "interrupts", None)
        for it in _interrupt_items_from_field(ints):
            val = getattr(it, "value", None)
            if val is None and isinstance(it, dict):
                val = it
            if isinstance(val, dict):
                return val
            if val is not None:
                return {"phase": "raw", "value": val}
    except Exception:
        return None
    return None


def is_game_finished(state: GameState | None) -> bool:
    if not state:
        return False
    return state.get("game_status") == "end"
