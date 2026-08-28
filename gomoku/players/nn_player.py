"""神经网络棋手：加载训练产出的 GomokuNet checkpoint，用 policy 头选点。

用法（GUI 对战）:
    python gui.py --model runs/smoke/ckpt_ep0_shard4.pt

用法（批量胜率）:
    python evaluate.py --p1 greedy --p2 nn --ckpt runs/smoke/ckpt_ep0_shard4.pt --games 20
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from gomoku.board import Board
from training.model import GomokuNet


class NNetPlayer:
    """输入 (1,2,N,N): 通道0=己方棋子, 通道1=对方。policy 头取 logits。

    temperature=0 时贪心选最大 logit（并列取近中心点），>0 时按 softmax 采样。
    选点范围限制在已有棋子 radius 邻域内（空盘落中心），跑得更快也更合理。
    """

    def __init__(self, color: int, ckpt: str, device: str | None = None,
                 temperature: float = 0.0, radius: int = 2, name: str = "nn"):
        self.color = color
        self.temperature = temperature
        self.radius = radius
        self.name = name

        dev = device or self._pick_device()
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        args = ck.get("args") or {}
        blocks = int(args.get("blocks", 8))
        channels = int(args.get("channels", 128))
        policy_out = int(args.get("policy_out", 225))
        self.net = GomokuNet(board=15, in_channels=2, channels=channels,
                             blocks=blocks, policy_out=policy_out, value_out=3)
        self.net.load_state_dict(ck["model"])
        self.net.eval().to(dev)
        self.device = dev

    @staticmethod
    def _pick_device() -> str:
        """有 GPU 且当前 torch 构建包含该算力内核才用 CUDA，否则 CPU。
        注: cu130 构建不含 sm_61(Pascal)，但 cu126 构建含，按 get_arch_list 判断更准。"""
        if torch.cuda.is_available():
            try:
                cap = torch.cuda.get_device_capability(0)
                archs = torch.cuda.get_arch_list()
                if any(a.startswith(f"sm_{cap[0]}") for a in archs):
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    def choose_move(self, board: Board) -> tuple[int, int] | None:
        candidates = self._candidates(board)
        if not candidates:
            return None
        n = board.size
        x = torch.zeros(1, 2, n, n, dtype=torch.float32, device=self.device)
        for r, c, color in board.stones():
            ch = 0 if color == self.color else 1
            x[0, ch, r, c] = 1.0

        with torch.no_grad():
            logits, _ = self.net(x)
        logits = logits[0].float()
        mask = torch.full((n * n,), float("-inf"), dtype=logits.dtype, device=self.device)
        for r, c in candidates:
            mask[r * n + c] = logits[r * n + c]

        if self.temperature > 0:
            probs = F.softmax(mask / max(self.temperature, 1e-3), dim=0)
            idx = torch.multinomial(probs, 1).item()
        else:
            best = mask.argmax().item()
            mx = mask[best].item()
            tied = (mask == mx).nonzero(as_tuple=False).flatten().tolist()
            hw = (n - 1) / 2
            idx = min(tied, key=lambda i: (abs(i // n - hw) ** 2 + abs(i % n - hw) ** 2))
        return divmod(idx, n)

    def _candidates(self, board: Board) -> list[tuple[int, int]]:
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
