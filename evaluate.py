"""棋手对局评估: 两个棋手对战统计胜率。

用法:
    python evaluate.py --p1 greedy --p2 random --games 100
    python evaluate.py --p1 greedy --p2 greedy --games 20 --seed 7
    python evaluate.py --p1 greedy --p2 random --games 100 --show-board
"""
from __future__ import annotations

import argparse
import random

from gomoku.board import Board
from gomoku.players.greedy import GreedyPlayer
from gomoku.players.random import RandomPlayer
from gomoku.players.point15_player import Point15Player

PLAYERS = {
    "greedy": GreedyPlayer,
    "random": RandomPlayer,
    "point15": Point15Player,
}


def play_once(cls1, cls2, size: int, seed: int, first: int):
    """first: 1=p1执黑先手, 2=p2执黑先手。返回 (胜方颜色, 手数)。"""
    rng = random.Random(seed)
    p1 = cls1(color=Board.BLACK if first == 1 else Board.WHITE)
    p2 = cls2(color=Board.WHITE if first == 1 else Board.BLACK)
    board = Board(size)
    turn = Board.BLACK
    while not board.is_full():
        player = p1 if turn == p1.color else p2
        mv = player.choose_move(board)
        if mv is None:
            break
        board.place(*mv, turn)
        w = board.winner()
        if w:
            return w, len(board.history)
        turn = 3 - turn
    return 0, len(board.history)


def main():
    ap = argparse.ArgumentParser(description="棋手对局评估")
    ap.add_argument("--p1", choices=list(PLAYERS), default="greedy")
    ap.add_argument("--p2", choices=list(PLAYERS), default="random")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    w1 = w2 = draw = 0
    total = 0
    for g in range(args.games):
        first = 1 if g % 2 == 0 else 2        # 黑白轮流坐庄
        winner, moves = play_once(PLAYERS[args.p1], PLAYERS[args.p2],
                                  args.size, args.seed * 10000 + g, first)
        total += moves
        if winner == 0:
            draw += 1
        elif (winner == Board.BLACK) == (first == 1):
            w1 += 1
        else:
            w2 += 1

    print(f"{args.p1} vs {args.p2}  x {args.games} 局 (15x15)")
    print(f"  {args.p1:<8}: {w1:>4} 胜 ({w1/args.games:.1%})")
    print(f"  {args.p2:<8}: {w2:>4} 胜 ({w2/args.games:.1%})")
    print(f"  平局          : {draw:>4}   平均手数 {total/args.games:.1f}")


if __name__ == "__main__":
    main()
