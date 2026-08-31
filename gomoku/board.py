"""15×15 五子棋纯逻辑引擎（无 GUI、无搜索）。

规则: 自由规则(Freestyle) —— 连五及以上(长连)即胜。
"""
from __future__ import annotations


class Board:
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(self, size: int = 15):
        self.size = size
        self.grid = [[0] * size for _ in range(size)]
        self.history: list[tuple[int, int, int]] = []  # (r, c, color)

    # ---- 查询 ----
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def get(self, r: int, c: int) -> int:
        return self.grid[r][c]

    def is_empty(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.grid[r][c] == Board.EMPTY

    def is_full(self) -> bool:
        return len(self.history) == self.size * self.size

    def legal_moves(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.grid[r][c] == Board.EMPTY]

    def stones(self) -> list[tuple[int, int, int]]:
        return list(self.history)

    # ---- 落子/撤销 ----
    def place(self, r: int, c: int, color: int) -> bool:
        if not self.is_empty(r, c):
            return False
        self.grid[r][c] = color
        self.history.append((r, c, color))
        return True

    def undo(self) -> bool:
        if not self.history:
            return False
        r, c, _ = self.history.pop()
        self.grid[r][c] = Board.EMPTY
        return True

    def clone(self) -> "Board":
        b = Board(self.size)
        b.grid = [row[:] for row in self.grid]
        b.history = list(self.history)
        return b

    def step(self, r: int, c: int, color: int) -> "Board":
        """返回落子后的新棋盘(不修改自身)。"""
        b = self.clone()
        b.place(r, c, color)
        return b

    # ---- 胜负 ----
    def winner(self) -> int:
        """返回胜方颜色 (1/2)，无胜方返回 0。长连(>=5)算胜。"""
        size = self.size
        grid = self.grid
        for r in range(size):
            for c in range(size):
                color = grid[r][c]
                if color == 0:
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    pr, pc = r - dr, c - dc
                    if self.in_bounds(pr, pc) and grid[pr][pc] == color:
                        continue  # 只从每条线的起点计数
                    cnt = 0
                    rr, cc = r, c
                    while self.in_bounds(rr, cc) and grid[rr][cc] == color:
                        cnt += 1
                        rr += dr
                        cc += dc
                    if cnt >= 5:
                        return color
        return 0
