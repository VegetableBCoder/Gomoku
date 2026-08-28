"""用旧版 point.py(15×15 版) 做成的棋手 —— 只用来当对照/基准。
"""
from __future__ import annotations

from gomoku import point15
from gomoku.board import Board


class Point15Player:
    def __init__(self, color: int = Board.BLACK, attack_weight: float = 1.05,
                 name: str = "point15"):
        self.color = color
        self.attack_weight = attack_weight
        self.name = name

    def choose_move(self, board: Board) -> tuple[int, int] | None:
        size = board.size
        record = [[board.get(r, c) for c in range(size)] for r in range(size)]
        opp = 3 - self.color
        best = None
        best_score = -1.0
        best_d2 = float("inf")
        hw = (size - 1) / 2
        for r in range(size):
            for c in range(size):
                if record[r][c] != Board.EMPTY:
                    continue
                attack = point15.Point(record, r, c, self.color).getGrade()
                defense = point15.Point(record, r, c, opp).getGrade()
                score = self.attack_weight * attack + defense
                d2 = (r - hw) ** 2 + (c - hw) ** 2
                if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and d2 < best_d2):
                    best_score, best_d2, best = score, d2, (r, c)
        return best
