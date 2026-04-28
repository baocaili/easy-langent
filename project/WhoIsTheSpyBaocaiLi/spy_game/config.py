from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMConfig:
    """大模型参数（可由 Streamlit 覆盖 .env）。"""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 800


@dataclass
class GameConfig:
    """对局规则：4~10 人；人数 > 6 时可选双卧底；无白板角色。"""

    num_players: int = 5
    num_undercover: int = 1
    human_player_id: Optional[str] = None
    player_name_prefix: str = "P"
    max_rounds: int = 30

    def __post_init__(self) -> None:
        if self.num_players < 4 or self.num_players > 10:
            raise ValueError("人数需在 4~10 之间")
        if self.num_players <= 6:
            self.num_undercover = 1
        else:
            if self.num_undercover not in (1, 2):
                raise ValueError("人数大于 6 时，卧底数量仅可为 1 或 2")
        if self.human_player_id is not None and self.human_player_id not in self.player_ids():
            raise ValueError("人类玩家 ID 不在本局玩家列表中")

    def player_ids(self) -> List[str]:
        return [f"{self.player_name_prefix}{i + 1}" for i in range(self.num_players)]


@dataclass
class RuntimeContext:
    """运行期上下文：注入节点闭包。"""

    llm: object
    parser: object
    game: GameConfig
    log: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)

    def append_log(self, line: str) -> None:
        self.log.append(line)

    def trace_node(self, name: str) -> None:
        self.trace.append(name)
