#!/bin/bash

# =================================================================
#  main.py 専用 実行スクリプト
# =================================================================
# このスクリプトは、WandBでの進捗確認とモデルの重み(.pt)の保存を
# 目的とした main.py を実行します。
# -----------------------------------------------------------------

# Python仮想環境を有効化
source venv/bin/activate

# --- 実験設定 (この部分を書き換えてください) ---

# LOSS: "ce" (クロスエントロピー) または "ls" (ラベルスムージング) を選択
LOSS="ls"

# EPS: LOSSが"ls"の場合のラベルスムージング強度 (例: "0.05")
#      LOSSが"ce"の場合は"0.0"のままにしてください
EPS="0.05"

# SAVE_INTERVAL: モデル(.pt)を何エポックごとに保存するか (0以下で無効化)
SAVE_INTERVAL=50

# --- ------------------------------------ ---

# 実験名を自動生成
EXP_NAME="${LOSS}${EPS}_main_run_B128"

echo "===== 実験を開始します: ${EXP_NAME} ====="
echo "使用スクリプト: main.py"

python main.py \
    --dset cifar10 \
    --model resnet18 \
    --wd 5e-4 \
    --scheduler ms \
    --max_epochs 800 \
    --batch_size 128 \
    --lr 0.05 \
    --log_freq 10 \
    --loss ${LOSS} \
    --eps ${EPS} \
    --seed 2021 \
    --exp_name ${EXP_NAME} \
    --save_ckpt ${SAVE_INTERVAL}

echo ""
echo "===== 実験が完了しました: ${EXP_NAME} ====="