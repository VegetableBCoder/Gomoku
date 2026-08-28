"""威胁阶梯贪心棋手（15×15），带有限 1-ply 前瞻。

决策: 对候选空位算「己方威胁等级」和「对方威胁等级」,
      score = attack_weight×己方 + 对方, 再对 base 分最高的 lookahead 个候选
      模拟对方最佳应手做惩罚, 避免走了就被反杀/被反威胁。
威胁等级(patterns.threat_metric): 连五 > 活四/双四 > 冲四/四三 > 双三 > 活三 > 发展分。
"""
from __future__ import annotations

from gomoku.board import Board
from gomoku.patterns import threat_metric


class GreedyPlayer:
    def __init__(self, color: int = Board.BLACK, attack_weight: float = 1.2,
                 radius: int = 2, lookahead: int = 8, reply_penalty: float = 0.3,
                 name: str = "greedy"):
        self.color = color
        self.attack_weight = attack_weight
        self.radius = radius
        self.lookahead = lookahead
        self.reply_penalty = reply_penalty
        self.name = name

    def choose_move(self, board: Board) -> tuple[int, int] | None:
        candidates = self._candidates(board)
        if not candidates:
            return None
        opp = 3 - self.color
        hw = (board.size - 1) / 2

        scored = []
        for r, c in candidates:
            attack = threat_metric(board, r, c, self.color)
            defense = threat_metric(board, r, c, opp)
            score = self.attack_weight * attack + defense
            scored.append((score, r, c, attack, defense))

        scored.sort(key=lambda t: -t[0])
        refined = []
        for score, r, c, attack, defense in scored[: self.lookahead]:
            b2 = board.step(r, c, self.color)
            replies = [
                self.attack_weight * threat_metric(b2, r2, c2, opp)
                + threat_metric(b2, r2, c2, self.color)
                for r2, c2 in self._candidates(b2)
            ]
            whats = max(replies) if replies else 0.0
            refined.append((score - self.reply_penalty * whats, r, c))
        pool = refined + [(t[0], t[1], t[2]) for t in scored[self.lookahead:]]

        best = pool[0][1:3]
        best_score, best_d2 = -1.0, float("inf")
        for score, r, c in pool:
            d2 = (r - hw) ** 2 + (c - hw) ** 2
            if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and d2 < best_d2):
                best_score, best_d2, best = score, d2, (r, c)
        return best

    def _candidates(self, board: Board) -> list[tuple[int, int]]:
        """空盘回中心格；否则取已有棋子邻居 radius 内的空位。"""
        size = board.size
        stones = [(r, c) for r in range(size) for c in range(size)
                  if board.get(r, c) != Board.EMPTY]
        if not stones:
            return [(size // 2, size // 2)]
        cand = set()
        for r, c in stones:
            for dr in range(-self.radius, self.radius + 1):
                for dc in range(-self.radius, self.radius + 1):
                    rr, cc = r + dr, c + dc
                    if board.is_empty(rr, cc):
                        cand.add((rr, cc))
        return sorted(cand)
