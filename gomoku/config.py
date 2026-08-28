"""项目配置：从项目根目录 .env 读取路径等运行配置（无额外依赖）。

优先级: 进程环境变量 > 项目根目录 .env 文件 > 内置默认值。
所有相对路径均以项目根目录（本文件向上一级）为基准，因此从任意目录
运行脚本都能拿到一致的结果。.env 不入库（见 .gitignore），给别人用
只需复制 .env.example 为 .env 并改成自己的路径。
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(root: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    p = root / ".env"
    if p.is_file():
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_OVERRIDES = _load_dotenv(_ROOT)


def get(key: str, default: str) -> str:
    if key in os.environ:
        return os.environ[key]
    return _OVERRIDES.get(key, default)


def path(key: str, default: str) -> Path:
    """绝对路径；相对值以项目根目录为基准。"""
    v = get(key, default)
    p = Path(v).expanduser()
    if not p.is_absolute():
        p = _ROOT / p
    return p


def project_root() -> Path:
    return _ROOT


def data_root() -> Path:
    """原始数据下载根目录（download 脚本的 --target）。"""
    return path("GOBANG_DATA_ROOT", "data")


def raw_dir() -> Path:
    """原始 ModelScope 数据子集目录。"""
    return path("GOBANG_RAW_DIR", "data/fs15x_label28b")


def processed_dir() -> Path:
    """预处理后的训练数据目录（train/val 分片）。"""
    return path("GOBANG_PROCESSED_DIR", "data/processed")


def runs_dir() -> Path:
    return path("GOBANG_RUNS_DIR", "runs")


def model_dir() -> Path:
    return path("GOBANG_MODEL_DIR", "models")
