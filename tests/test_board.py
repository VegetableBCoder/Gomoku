"""引擎与贪心棋手的基础测试:  python -m tests.test_board"""
from gomoku.board import Board
from gomoku.players.greedy import GreedyPlayer
from gomoku.players.random import RandomPlayer


def test_win_lines():
    # 横五
    b = Board()
    for c in range(5, 10):
        assert b.place(7, c, Board.BLACK)
    assert b.winner() == Board.BLACK, "横五未判胜"
    # 竖五
    b = Board()
    for r in range(3, 8):
        assert b.place(r, 4, Board.WHITE)
    assert b.winner() == Board.WHITE, "竖五未判胜"
    # 主对角线
    b = Board()
    for i in range(5):
        assert b.place(2 + i, 2 + i, Board.BLACK)
    assert b.winner() == Board.BLACK, "斜五未判胜"
    # 副对角线
    b = Board()
    for i in range(5):
        assert b.place(2 + i, 12 - i, Board.WHITE)
    assert b.winner() == Board.WHITE, "反斜五未判胜"
    # 长连也算胜(自由规则)
    b = Board()
    for c in range(9, 15):   # 合法列 0~14, 6 连即长连
        assert b.place(7, c, Board.BLACK)
    assert b.winner() == Board.BLACK, "长连未判胜"


def test_no_win():
    b = Board()
    for c in range(5, 9):          # 只有 4 连
        b.place(7, c, Board.BLACK)
    assert b.winner() == 0, "4连不应判胜"


def test_place_undo_illegal():
    b = Board()
    assert b.place(3, 3, Board.BLACK)
    assert not b.place(3, 3, Board.WHITE), "重复落子应失败"
    assert not b.place(-1, 0, Board.BLACK), "越界应失败"
    assert b.undo()
    assert b.get(3, 3) == Board.EMPTY


def test_greedy_basics():
    b = Board()
    p = GreedyPlayer(color=Board.BLACK)
    mv = p.choose_move(b)
    assert mv == (7, 7), f"空盘贪心应下中心, got {mv}"
    # 黑四连 + 一步活: 贪心应立即成五
    b = Board()
    for c in range(5, 9):
        b.place(7, c, Board.BLACK)
    mv = p.choose_move(b)
    assert mv in ((7, 4), (7, 9)), f"应补五, got {mv}"
    assert b.step(*mv, Board.BLACK).winner() == Board.BLACK


def test_random_legal():
    b = Board()
    p = RandomPlayer(color=Board.WHITE, seed=1)
    for _ in range(10):
        mv = p.choose_move(b)
        assert mv is not None and b.is_empty(*mv)


if __name__ == "__main__":
    test_win_lines()
    test_no_win()
    test_place_undo_illegal()
    test_greedy_basics()
    test_random_legal()
    print("全部测试通过 ✅")
