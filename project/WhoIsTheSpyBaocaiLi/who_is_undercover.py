"""控制台入口：全 AI 观战一局（无人类 interrupt，一次跑完）。"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from spy_game.config import GameConfig, RuntimeContext
from spy_game.llm_factory import load_llm
from spy_game.runner import build_app, start_new_game


def main() -> None:
    _root = Path(__file__).resolve().parent
    load_dotenv(_root / ".env")
    load_dotenv(_root.parent / ".env")
    llm, parser = load_llm()
    game = GameConfig(num_players=5, num_undercover=1, human_player_id=None)
    ctx = RuntimeContext(llm=llm, parser=parser, game=game)
    app = build_app(ctx)
    print("=" * 50)
    print("谁是卧底 · 全 AI 观战模式")
    print("=" * 50)
    state, _config = start_new_game(app, ctx, game.player_ids())
    print("\n".join(ctx.log))
    print("\n--- 最终状态 ---")
    print(f"status={state.get('game_status')} winner={state.get('winner')}")
    print(f"平民词={state.get('civilian_word')} 卧底词={state.get('undercover_word')}")
    print(f"淘汰顺序={state.get('eliminated')}")


if __name__ == "__main__":
    main()
