from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict


RoleWord = Tuple[str, str]


class GameState(TypedDict, total=False):
    civilian_word: str
    undercover_word: str
    players: List[str]
    role_assignment: Dict[str, RoleWord]
    speeches: Dict[str, str]
    history_speeches: List[Dict[str, str]]
    speech_reasoning: Dict[str, str]
    votes: Dict[str, str]
    vote_reasoning: Dict[str, str]
    game_status: str
    winner: str
    eliminated: List[str]
    round: int
    speech_order: List[str]
    speech_idx: int
    vote_order: List[str]
    vote_idx: int
    max_rounds: int


def empty_state(players: List[str]) -> GameState:
    return {
        "civilian_word": "",
        "undercover_word": "",
        "players": list(players),
        "role_assignment": {},
        "speeches": {},
        "history_speeches": [],
        "speech_reasoning": {},
        "votes": {},
        "vote_reasoning": {},
        "game_status": "running",
        "winner": "",
        "eliminated": [],
        "round": 1,
        "speech_order": [],
        "speech_idx": 0,
        "vote_order": [],
        "vote_idx": 0,
        "max_rounds": 30,
    }
