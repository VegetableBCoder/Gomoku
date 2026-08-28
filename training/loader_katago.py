"""流式读取预处理后的 KataGomo 分片 (processed/{train,val}/data*.npz)。

每个 npz 三个数组:
    board  (N, 2, 15, 15) uint8   己方/对方
    policy (N, 225)      int16    策略原始计数 (未归一化)
    value  (N, 3)        float16  胜率/负率/和棋率

load_shard() 返回:
    board  (N,2,15,15) float32   (兼容旧版预处理遗留的 (N,2,232) 形状, 自动 reshape)
    policy (N,225)     float32   已按行归一化成概率
    value  (N,3)       float32
纯 numpy 实现, 不依赖 torch, 可单独测试。
"""
from pathlib import Path
import numpy as np

BOARD = 15


class KatagoShards:
    def __init__(self, root: str, split: str = "train", limit: int = 0, seed: int = 0):
        files = sorted(Path(root).glob(f"{split}/data*.npz"))
        if limit:
            files = files[:limit]
        self.files = files
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.files)

    def shuffled_files(self):
        files = self.files[:]
        self.rng.shuffle(files)
        return files

    @staticmethod
    def load_shard(path):
        data = np.load(path)
        n = data["board"].shape[0]
        b = data["board"]
        if b.shape[2] == 232:
            # 旧版预处理的遗留格式: 每通道 232 位含 7 个 padding 位(全0), 只取前 225 位
            b = b[:, :, :225]
        board = b.reshape(n, 2, BOARD, BOARD).astype(np.float32)
        counts = data["policy"].astype(np.float32)            # (N,225) int16 计数
        sums = counts.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        policy = counts / sums                                # 归一化成概率
        value = data["value"].astype(np.float32)              # (N,3) 胜/负/和
        return board, policy, value
