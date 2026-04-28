from __future__ import annotations

import json
import random
import textwrap
from typing import Dict, List, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from spy_game.config import RuntimeContext
from spy_game.state_types import GameState
from spy_game.utils import strip_json_fences


def _invoke_llm_text(ctx: RuntimeContext, system: str, user: str) -> str:
    """直接用消息列表调用模型，避免 ChatPromptTemplate 把发言/JSON 里的花括号当成占位符。"""
    msgs: list[BaseMessage] = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    ai = ctx.llm.invoke(msgs)
    return str(ctx.parser.invoke(ai))


def _players(state: GameState) -> List[str]:
    if state.get("players"):
        return list(state["players"])
    return sorted(state.get("role_assignment", {}).keys())


def _alive(state: GameState) -> List[str]:
    return [p for p in _players(state) if p not in state.get("eliminated", [])]


def _counts(state: GameState, pool: List[str]) -> Tuple[int, int]:
    ra = state.get("role_assignment", {})
    civ = sum(1 for p in pool if ra[p][0] == "平民")
    uc = sum(1 for p in pool if ra[p][0] == "卧底")
    return civ, uc


def node_generate_words(state: GameState, ctx: RuntimeContext) -> dict:
    ctx.trace_node("generate_words")
    n = len(_players(state))
    # 不在模板里写带花括号的 JSON 示例，避免 LangChain 各版本对 {{ }} 解析不一致。
    system_template = """你是专业的「谁是卧底」游戏出题人，需生成一组高质量的词语对。
本局人数 {n} 人，请适配难度。
核心要求：
1. 词语类型：日常物品/食品/场景，避免生僻词
2. 语义关系：平民词与卧底词高度相似但核心特征不同
3. 输出格式：只输出一段可被解析的 JSON 文本，且仅包含两个英文键名：civilian（平民词）、undercover（卧底词），键名与中文词值均用英文双引号包裹；除这段 JSON 外不要输出任何其它字符。
禁止输出任何额外文字，只返回 JSON 字符串！"""
    # 先替换 {n}，使 system 中不再含任何花括号，彻底避免模板误解析
    system_fixed = system_template.replace("{n}", str(n))
    result = _invoke_llm_text(
        ctx,
        system_fixed,
        "生成一组符合要求的谁是卧底词语对",
    )
    try:
        word_data = json.loads(strip_json_fences(str(result)))
        civilian_word = word_data["civilian"]
        undercover_word = word_data["undercover"]
    except (json.JSONDecodeError, KeyError, TypeError):
        fallback_pairs = [
            ("奶茶", "果汁"),
            ("牙刷", "牙膏"),
            ("米饭", "面条"),
            ("手机", "平板"),
            ("篮球", "足球"),
            ("咖啡", "红茶"),
        ]
        civilian_word, undercover_word = random.choice(fallback_pairs)
    ctx.append_log(f"🎯 词语生成完成：平民词={civilian_word} ｜ 卧底词={undercover_word}")
    return {"civilian_word": civilian_word, "undercover_word": undercover_word}


def node_assign_roles(state: GameState, ctx: RuntimeContext) -> dict:
    ctx.trace_node("assign_roles")
    g = ctx.game
    players = list(state.get("players") or g.player_ids())
    uc_n = g.num_undercover
    uc_set = set(random.sample(players, uc_n))
    role_assignment: Dict[str, Tuple[str, str]] = {}
    for p in players:
        if p in uc_set:
            role_assignment[p] = ("卧底", state["undercover_word"])
        else:
            role_assignment[p] = ("平民", state["civilian_word"])
    ctx.append_log("🎭 角色分配完成")
    for a, (r, w) in role_assignment.items():
        ctx.append_log(f"  {a}：{r}（{w}）")
    return {"role_assignment": role_assignment, "players": players}


def node_begin_speech_round(state: GameState, ctx: RuntimeContext) -> dict:
    ctx.trace_node("begin_speech_round")
    cap = int(state.get("max_rounds", ctx.game.max_rounds))
    if int(state.get("round", 1)) > cap:
        ctx.append_log("⏱ 超过最大轮次上限，判定平民方胜利。")
        return {"game_status": "end", "winner": "civilian"}
    alive = _alive(state)
    order = list(alive)
    random.shuffle(order)
    ctx.append_log(f"📢 第{state.get('round', 1)}轮发言顺序：{' → '.join(order)}")
    return {
        "speeches": {},
        "speech_reasoning": {},
        "speech_order": order,
        "speech_idx": 0,
    }


def node_speech_turn(state: GameState, ctx: RuntimeContext) -> dict:
    ctx.trace_node("speech_turn")
    order = list(state.get("speech_order") or [])
    idx = int(state.get("speech_idx", 0))
    if idx >= len(order):
        return {}
    pid = order[idx]
    ra = state["role_assignment"]
    role, word = ra[pid]
    current_round = int(state.get("round", 1))

    history_context = ""
    if state.get("history_speeches"):
        history_context = "【历史发言记录】\n"
        for ridx, round_speeches in enumerate(state["history_speeches"], 1):
            history_context += f"第{ridx}轮发言：\n"
            for agent, speech in round_speeches.items():
                if agent not in state.get("eliminated", []):
                    history_context += f"- {agent}：{speech}\n"
        history_context += "\n"

    prev_same_round = ""
    speeches_so_far = dict(state.get("speeches") or {})
    if idx > 0:
        prev_same_round = "【本轮已发言】\n"
        for i in range(idx):
            q = order[i]
            if q in speeches_so_far:
                prev_same_round += f"- {q}：{speeches_so_far[q]}\n"
        prev_same_round += "\n"

    human_id = ctx.game.human_player_id
    if human_id == pid:
        payload = {
            "phase": "speech",
            "player": pid,
            "round": current_round,
            "role": role,
            "word": word,
            "hint": "请用一句话描述你的词，不要直接说出词语本身。",
        }
        raw = interrupt(payload)
        speech = raw if isinstance(raw, str) else str(raw.get("text", raw) if isinstance(raw, dict) else raw)
        reason = "（人类玩家发言）"
    else:
        system_speech = f"""你是「谁是卧底」游戏的资深玩家，当前是第{current_round}轮发言。
【核心规则】
1. 字数：10-100 个汉字（不含标点），语义完整。
2. 禁止直接说出词语本身。
3. 角色策略：
   - 平民：描述核心特征，协助找出卧底；避免与他人重复。
   - 卧底：模仿平民，模糊差异，不暴露。
4. 输出格式：只输出一段 JSON，含两个英文键 speech（你的发言正文）与 reason（简短策略说明），键与字符串值均用英文双引号；除该 JSON 外不要输出其它字符。
禁止输出任何额外文字。
{history_context}{prev_same_round}"""
        user_speech = "".join(["你的角色是", role, "，拿到的词语是", word])
        output = _invoke_llm_text(ctx, system_speech, user_speech)
        try:
            data = json.loads(strip_json_fences(str(output)))
            speech = str(data["speech"])
            reason = str(data.get("reason", ""))
            if len(speech) < 10:
                speech = speech + "，是生活中常见的东西，大家几乎都接触过。"
        except (json.JSONDecodeError, KeyError, TypeError):
            speech = f"第{current_round}轮：这是日常常见的东西，使用场景很多，特征比较明显。"
            reason = "模型输出解析失败，使用兜底发言。"
        ctx.append_log(f"  {pid}（{role}）发言：{speech}")

    speeches = dict(state.get("speeches") or {})
    speeches[pid] = speech
    reasoning = dict(state.get("speech_reasoning") or {})
    reasoning[pid] = reason
    return {
        "speeches": speeches,
        "speech_reasoning": reasoning,
        "speech_idx": idx + 1,
    }


def node_finalize_speeches(state: GameState, ctx: RuntimeContext) -> dict:
    ctx.trace_node("finalize_speeches")
    snap = dict(state.get("speeches") or {})
    hist = list(state.get("history_speeches") or [])
    # interrupt 后整节点会重跑：避免重复把同一轮发言压入 history
    if not hist or hist[-1] != snap:
        hist = [*hist, snap]
        ctx.append_log("🔒 本轮发言已归档，进入投票。")
    # 有人类玩家时先暂停在「发言汇总」，避免提交后直接跳进投票且看不到全员发言
    if ctx.game.human_player_id is not None:
        interrupt(
            {
                "phase": "recap",
                "round": int(state.get("round", 1)),
                "speech_order": list(state.get("speech_order") or []),
            }
        )
    return {"history_speeches": hist}


def node_vote_begin(state: GameState, ctx: RuntimeContext) -> dict:
    ctx.trace_node("vote_begin")
    alive = _alive(state)
    order = list(alive)
    random.shuffle(order)
    ctx.append_log(f"🗳 投票顺序：{' → '.join(order)}")
    return {"votes": {}, "vote_reasoning": {}, "vote_order": order, "vote_idx": 0}


def node_vote_order_preview(state: GameState, ctx: RuntimeContext) -> dict:
    """人类入局时：在首轮投票前暂停，仅展示顺序（无模型调用），避免与「发言汇总」叠在一起时显得卡顿来源不明。"""
    ctx.trace_node("vote_order_preview")
    if ctx.game.human_player_id is None:
        return {}
    order = list(state.get("vote_order") or [])
    hid = ctx.game.human_player_id
    try:
        hi = order.index(hid)
    except ValueError:
        hi = -1
    interrupt(
        {
            "phase": "vote_order",
            "round": int(state.get("round", 1)),
            "order": order,
            "human_index": hi,
            "ai_votes_before_human": max(0, hi),
        }
    )
    return {}


def node_vote_turn(state: GameState, ctx: RuntimeContext) -> dict:
    ctx.trace_node("vote_turn")
    order = list(state.get("vote_order") or [])
    idx = int(state.get("vote_idx", 0))
    if idx >= len(order):
        return {}
    voter = order[idx]
    ra = state["role_assignment"]
    role, word = ra[voter]
    current_round = int(state.get("round", 1))
    alive = _alive(state)
    targets = [p for p in alive if p != voter]

    speech_context = f"【第{current_round}轮发言】\n"
    for agent, speech in (state.get("speeches") or {}).items():
        speech_context += f"{agent}：{speech}\n"
    if state.get("history_speeches"):
        speech_context += "\n【历史发言参考】\n"
        for hidx, round_speeches in enumerate(state["history_speeches"][:-1], 1):
            speech_context += f"第{hidx}轮：\n"
            for agent, speech in round_speeches.items():
                if agent in alive:
                    speech_context += f"- {agent}：{speech}\n"

    human_id = ctx.game.human_player_id
    if human_id == voter:
        raw = interrupt(
            {
                "phase": "vote",
                "voter": voter,
                "round": current_round,
                "targets": targets,
                "hint": "请选择要投票淘汰的玩家 ID。",
            }
        )
        if isinstance(raw, dict):
            vote = str(raw.get("target") or raw.get("vote") or "")
        else:
            vote = str(raw).strip()
        reason = "（人类玩家投票）"
    else:
        system_vote = (
            """你是「谁是卧底」游戏的理性玩家，需基于当前轮与历史发言投票。
【规则】
1. 平民：找出描述异常、前后矛盾的玩家。
2. 卧底：投票给看起来像平民的玩家，避免暴露自己。
3. 输出格式：只输出一段 JSON，含英文键 vote（你要投给的玩家 ID，如 P2）与 reason（投票理由）；vote 必须是可选对象之一。除该 JSON 外不要输出其它字符。
禁止输出任何额外文字。

"""
            + speech_context
        )
        user_vote = "\n".join(
            [
                "你的角色：" + role,
                "你的词语：" + word,
                "可选投票对象：" + ", ".join(targets),
            ]
        )
        output = _invoke_llm_text(ctx, system_vote, user_vote)
        try:
            data = json.loads(strip_json_fences(str(output)))
            vote = str(data["vote"]).strip()
            reason = str(data.get("reason", ""))
        except (json.JSONDecodeError, KeyError, TypeError):
            vote = random.choice(targets)
            reason = textwrap.shorten("解析失败，随机投票。", width=50)

    if vote == voter or vote not in alive:
        vote = random.choice(targets)

    votes = dict(state.get("votes") or {})
    votes[voter] = vote
    reasons = dict(state.get("vote_reasoning") or {})
    reasons[voter] = reason
    ctx.append_log(f"  {voter} 投给 {vote}：{reason}")
    return {"votes": votes, "vote_reasoning": reasons, "vote_idx": idx + 1}


def node_judge_result(state: GameState, ctx: RuntimeContext) -> dict:
    ctx.trace_node("judge_result")
    rnd = int(state.get("round", 1))

    votes = state.get("votes") or {}
    vote_count: Dict[str, int] = {}
    for v in votes.values():
        vote_count[v] = vote_count.get(v, 0) + 1
    max_vote = max(vote_count.values()) if vote_count else 0
    tied = [a for a, c in vote_count.items() if c == max_vote]
    eliminated = random.choice(tied)
    eliminated_list = list(state.get("eliminated") or [])
    eliminated_list.append(eliminated)
    role_elim = state["role_assignment"][eliminated][0]
    ctx.append_log(f"❌ 第{rnd}轮淘汰：{eliminated}（{role_elim}），平票时在最高票中随机。")

    remaining = [p for p in _players(state) if p not in eliminated_list]
    civ, uc = _counts(state, remaining)

    if role_elim == "卧底":
        if uc == 0:
            ctx.append_log("🎉 平民胜利！所有卧底已出局。")
            return {
                "eliminated": eliminated_list,
                "game_status": "end",
                "winner": "civilian",
            }
    else:
        if civ == 1 and uc >= 1:
            ctx.append_log("🎉 卧底胜利！剩余 1 平民与至少 1 名卧底。")
            return {
                "eliminated": eliminated_list,
                "game_status": "end",
                "winner": "undercover",
            }

    ctx.append_log(f"➡ 游戏继续，进入第{rnd + 1}轮。")
    return {
        "eliminated": eliminated_list,
        "game_status": "running",
        "round": rnd + 1,
    }


def node_show_final_result(state: GameState, ctx: RuntimeContext) -> dict:
    ctx.trace_node("show_final_result")
    w = state.get("winner", "")
    ctx.append_log("=" * 40)
    ctx.append_log("📜 游戏结束")
    ctx.append_log(f"胜利方：{'平民' if w == 'civilian' else '卧底'}")
    ctx.append_log(f"平民词：{state.get('civilian_word','')} | 卧底词：{state.get('undercover_word','')}")
    ctx.append_log(f"总轮次：{state.get('round', 1)} | 淘汰顺序：{state.get('eliminated', [])}")
    ctx.append_log("=" * 40)
    return {}


def route_after_speech(state: GameState) -> str:
    order = state.get("speech_order") or []
    idx = int(state.get("speech_idx", 0))
    return "speech_turn" if idx < len(order) else "finalize_speeches"


def route_after_vote(state: GameState) -> str:
    order = state.get("vote_order") or []
    idx = int(state.get("vote_idx", 0))
    return "vote_turn" if idx < len(order) else "judge_result"


def route_after_judge(state: GameState) -> str:
    return "begin_speech_round" if state.get("game_status") == "running" else "show_final_result"


def route_after_begin(state: GameState) -> str:
    """超时或其它原因在 begin 内结束对局时直接进入结算。"""
    if state.get("game_status") == "end":
        return "show_final_result"
    return "speech_turn"
