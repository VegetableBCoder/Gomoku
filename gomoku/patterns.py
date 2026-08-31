"""棋型评估: 给「在空位 (r,c) 落 color」打分（贪心棋手核心）。"""
from __future__ import annotations

from gomoku.board import Board

WIN = 1_000_000        # 连五
OPEN_FOUR = 100_000    # 活四（双端开口连四 / 双冲四）
RUSH_FOUR = 10_000     # 冲四 / 跳四（一步可成五的威胁）
OPEN_THREE = 5_000     # 活三
THREE = 800            # 眠三 / 冲三
OPEN_TWO = 200         # 活二
TWO = 40               # 眠二
SINGLE = 2

_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def evaluate_place(board: Board, r: int, c: int, color: int) -> int:
    """在空位 (r,c) 落 color 后，四个方向棋型分之和（双威胁自动叠加）。"""
    if not board.is_empty(r, c):
        return 0
    total = 0
    for dr, dc in _DIRS:
        total += _line_score(board, r, c, color, dr, dc)
    return total


def direction_values(board: Board, r: int, c: int, color: int) -> list[int]:
    """四个方向各自的棋型分。"""
    if not board.is_empty(r, c):
        return [0] * 4
    return [_line_score(board, r, c, color, dr, dc) for dr, dc in _DIRS]


def threat_metric(board: Board, r: int, c: int, color: int) -> int:
    """威胁阶梯评分（贪心棋手决策用）:

    连五 > 活四/双四 > 冲四/四三 > 双三 > 活三 > 普通发展分。
    按「威胁等级」而不是数值相加排序，可避免单威胁被无限互堵。
    """
    vals = direction_values(board, r, c, color)
    win = any(v >= WIN for v in vals)
    fours = sum(1 for v in vals if v >= RUSH_FOUR)      # 活四/冲四/跳四
    threes = sum(1 for v in vals if OPEN_THREE <= v < RUSH_FOUR)  # 活三
    if win:
        return 100_000_000
    if fours >= 2:
        return 50_000_000
    if fours >= 1 and threes >= 1:
        return 40_000_000                               # 四三
    if fours >= 1:
        return 30_000_000                               # 单四(含冲四)
    if threes >= 2:
        return 25_000_000                               # 双三 ⭐
    if threes >= 1:
        return 10_000_000
    return sum(vals)                                    # 平时: 用细分数发展



def _line_score(board: Board, r: int, c: int, color: int, dr: int, dc: int) -> int:
    """沿一个方向统计连段长度与两端开口，并检查隔一子的跳型。"""
    run = 1
    # 正方向
    rr, cc = r + dr, c + dc
    while board.in_bounds(rr, cc) and board.get(rr, cc) == color:
        run += 1
        rr += dr
        cc += dc
    open_f = board.in_bounds(rr, cc) and board.get(rr, cc) == Board.EMPTY
    end_f = (rr, cc)
    # 反方向
    r2, c2 = r - dr, c - dc
    while board.in_bounds(r2, c2) and board.get(r2, c2) == color:
        run += 1
        r2 -= dr
        c2 -= dc
    open_b = board.in_bounds(r2, c2) and board.get(r2, c2) == Board.EMPTY
    end_b = (r2, c2)

    score = _base_score(run, open_f, open_b)
    if score >= WIN:          # 连五就是连五, 不要再叠加其他分支分
        return score
    if open_f:
        score += _gap_bonus(board, color, dr, dc, run, end_f)
    if open_b:
        score += _gap_bonus(board, color, -dr, -dc, run, end_b)
    return score


def _base_score(run: int, open_f: bool, open_b: bool) -> int:
    opens = open_f + open_b
    if run >= 5:
        return WIN
    if run == 4:
        if opens == 2:
            return OPEN_FOUR
        if opens == 1:
            return RUSH_FOUR
        return 0
    if run == 3:
        if opens == 2:
            return OPEN_THREE
        if opens == 1:
            return THREE
        return 0
    if run == 2:
        if opens == 2:
            return OPEN_TWO
        if opens == 1:
            return TWO
        return 0
    return SINGLE if opens else 0


def _gap_bonus(board: Board, color: int, dr: int, dc: int, run: int,
               end: tuple[int, int]) -> int:
    """开放端外侧隔一格还有己方子 → 跳四/跳三威胁。"""
    rr, cc = end[0] + dr, end[1] + dc       # 隔一空的落点
    if not board.in_bounds(rr, cc) or board.get(rr, cc) != color:
        return 0
    rr2, cc2 = rr + dr, cc + dc             # 它后面还要有发展空间
    if not board.in_bounds(rr2, cc2) or board.get(rr2, cc2) != Board.EMPTY:
        return 0
    run2 = 1
    rr3, cc3 = rr2 + dr, cc2 + dc
    while board.in_bounds(rr3, cc3) and board.get(rr3, cc3) == color:
        run2 += 1
        rr3 += dr
        cc3 += dc
    total = run + run2                       # 隔一空的两段长度和
    if total >= 5:
        return RUSH_FOUR                     # XXXX_ X 型: 补空即五
    if total == 4:
        return RUSH_FOUR                     # XX_XX / XXX_X 型: 跳四
    if total == 3:
        return OPEN_THREE // 5               # 跳活三(约1000)
    return 0
