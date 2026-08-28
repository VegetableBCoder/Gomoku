#!/usr/bin/env python3
"""15×15 五子棋 GUI：人 vs 贪心棋手。

自绘棋盘 + 使用 legacy/black.png、legacy/white.png 作为棋子。
运行:  python gui.py      (需要 tkinter: sudo apt install python3-tk)
"""
from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

from gomoku.board import Board
from gomoku.players.greedy import GreedyPlayer

N = 15
CELL = 36
MARGIN = 28
CANVAS = MARGIN * 2 + (N - 1) * CELL        # 560
STARS = [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11),
         (11, 3), (11, 7), (11, 11)]


class GomokuApp:
    def __init__(self, root: tk.Tk, make_engine=None):
        self.root = root
        root.title("瓜皮五子棋 (15×15)")
        self.board = Board(N)
        self.after_id = None
        self.locked = False
        # make_engine(color) -> 棋手实例；默认贪心，--model 时换神经网络
        self.make_engine = make_engine or (lambda color: GreedyPlayer(color=color))

        base = Path(__file__).resolve().parent / "legacy"
        self.black_img = self._load(base / "black.png")
        self.white_img = self._load(base / "white.png")

        top = tk.Frame(root)
        top.pack(pady=6)
        tk.Label(top, text="执子: ").pack(side="left")
        self.var = tk.StringVar(value="黑")
        for t in ("黑", "白"):
            tk.Radiobutton(top, text=t, variable=self.var, value=t,
                           command=self.new_game).pack(side="left", padx=4)
        tk.Button(top, text="新对局", command=self.new_game).pack(side="left", padx=10)
        tk.Button(top, text="悔棋", command=self.undo).pack(side="left", padx=2)

        self.status = tk.Label(root, text="", font=("Helvetica", 12), pady=4)
        self.status.pack()

        self.cv = tk.Canvas(root, width=CANVAS, height=CANVAS,
                            bg="#dcb35c", highlightthickness=0)
        self.cv.pack(padx=8, pady=4)
        self.cv.bind("<Button-1>", self.on_click)

        self.pos = {}
        for r in range(N):
            for c in range(N):
                self.pos[(r, c)] = (MARGIN + c * CELL, MARGIN + r * CELL)

        self.new_game()

    # ---------- 基础 ----------
    def _load(self, path: Path):
        img = Image.open(path).resize((CELL - 6, CELL - 6), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    def human_color(self) -> int:
        return Board.BLACK if self.var.get() == "黑" else Board.WHITE

    def draw_board(self):
        cv = self.cv
        cv.delete("all")
        for i in range(N):
            cv.create_line(MARGIN, MARGIN + i * CELL,
                           MARGIN + (N - 1) * CELL, MARGIN + i * CELL)
            cv.create_line(MARGIN + i * CELL, MARGIN,
                           MARGIN + i * CELL, MARGIN + (N - 1) * CELL)
        for r, c in STARS:
            x, y = self.pos[(r, c)]
            cv.create_oval(x - 4, y - 4, x + 4, y + 4, fill="black")
        for r, c, color in self.board.stones():
            x, y = self.pos[(r, c)]
            cv.create_image(x, y,
                            image=self.black_img if color == Board.BLACK else self.white_img)

    def winner(self):
        return self.board.winner()

    def refresh(self):
        self.draw_board()
        w = self.winner()
        if w:
            self.locked = True
            self.status.config(text="黑棋胜！" if w == Board.BLACK else "白棋胜！")
            messagebox.showinfo("胜负", "黑棋胜！" if w == Board.BLACK else "白棋胜！")
        elif self.board.is_full():
            self.locked = True
            self.status.config(text="平局")
        else:
            nxt = Board.BLACK if len(self.board.history) % 2 == 0 else Board.WHITE
            who = "你" if nxt == self.human_color() else "电脑"
            self.status.config(text=f"轮到 {who}（{'黑' if nxt == Board.BLACK else '白'}）")

    # ---------- 交互 ----------
    def on_click(self, event):
        if self.locked:
            return
        if len(self.board.history) % 2 != 0:     # 电脑刚下完，等你
            return
        col = round((event.x - MARGIN) / CELL)
        row = round((event.y - MARGIN) / CELL)
        if not self.board.is_empty(row, col):
            return
        self.board.place(row, col, self.human_color())
        self.refresh()
        if self.winner() or self.board.is_full():
            return
        self.locked = True
        self.status.config(text="电脑思考中…")
        self.after_id = self.root.after(150, self.computer_move)

    def computer_move(self):
        self.after_id = None
        comp = 3 - self.human_color()
        mv = self.make_engine(comp).choose_move(self.board)
        if mv is not None:
            self.board.place(*mv, comp)
        self.locked = False
        self.refresh()

    def new_game(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.board = Board(N)
        self.locked = False
        self.refresh()
        if self.human_color() == Board.WHITE:      # 电脑执黑先手
            self.locked = True
            self.after_id = self.root.after(200, self.computer_move)

    def undo(self):
        if self.locked:
            return
        if len(self.board.history) >= 2:
            self.board.undo()
            self.board.undo()
            self.refresh()


def main():
    ap = argparse.ArgumentParser(description="人机五子棋 (电脑默认=贪心)")
    ap.add_argument("--model", metavar="CKPT.pt",
                    help="用神经网络 checkpoint 当电脑 (如 runs/smoke/ckpt_ep0_shard4.pt)")
    ap.add_argument("--device", default=None,
                    help="模型推理设备: cpu / cuda (默认自动选择)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="模型选点温度: 0=贪心取最大概率 (默认), >0=按概率采样")
    args = ap.parse_args()

    root = tk.Tk()
    if args.model:
        from gomoku.players.nn_player import NNetPlayer
        make_engine = lambda color: NNetPlayer(
            color, args.model, device=args.device, temperature=args.temperature)
    else:
        make_engine = lambda color: GreedyPlayer(color=color)
    GomokuApp(root, make_engine)
    root.mainloop()


if __name__ == "__main__":
    main()
