"""随机棋手: 基线对手。"""
from __future__ import annotations

import random

from gomoku.board import Board


class RandomPlayer:
    def __init__(self, color: int = Board.BLACK, seed: int | None = None,
                 name: str = "random"):
        self.color = color
        self.name = name
        self.rng = random.Random(seed)

    def choose_move(self, board: Board) -> tuple[int, int] | None:
        moves = board.legal_moves()
        return self.rng.choice(moves) if moves else None
