# 瓜皮五子棋 2.0

纯逻辑引擎 + 贪心棋手 + KataGo 蒸馏数据训练。

## 目录结构

    gomoku/              核心包（纯逻辑，无 GUI）
      board.py           15×15 棋盘引擎：落子/撤销/胜负(长连算胜)/合法位
      patterns.py        棋型评估：活四/冲四/活三/眠三/活二… + 跳型
      players/
        greedy.py        贪心棋手（进攻×1.05 + 防守，一步杀/堵杀）
        random.py        随机棋手（对战基线）
    training/
      model.py           8×128 残差 CNN 双头（policy+value, ~2.4M 参数）
      loader_katago.py   KataGo 蒸馏数据流式加载
      train.py           监督训练 CLI
    scripts/
      download_ms_dataset.py   下载 ModelScope katago-gomoku-distill fs15x
      preprocess_katago.py     原始 npz → 精简训练格式
    tests/               引擎/棋手单测
    legacy/              旧版 tkinter GUI + point.py（2017 年作品，保留）
    evaluate.py          棋手对战评估 CLI
    runs/                训练输出（gitignore）

## 快速上手

    # 对战评估（不需要 GPU）
    python evaluate.py --p1 greedy --p2 random --games 100

    # 模型冒烟（CPU 可跑）
    python -m training.train --data /home/kita-ikuyo/dataset/processed \
        --limit-shards 5 --val-shards 3 --epochs 1 --batch-size 256 --device cpu \
        --out runs/smoke

    # 全量训练（3060 GPU）
    python -m training.train --data /home/kita-ikuyo/dataset/processed \
        --epochs 2 --batch-size 1024 --amp --out runs/full

    # 旧版 GUI
    cd legacy && python test.py

## 依赖

    uv pip install numpy            # 核心
    uv pip install torch            # 训练（cu130; 1060 用 CPU：--device cpu）
    uv pip install pillow           # 旧版 GUI 用
