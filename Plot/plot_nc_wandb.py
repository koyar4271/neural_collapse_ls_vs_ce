import os, pickle
import torch, io
from matplotlib import pyplot as plt
import numpy as np
from Plot.utils import add_headers
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
path = 'results_0/wandb'

# ============================================== plot ==============================================
mosaic = [
    ["A0", "A1", "A2", "A3"],
    ["B0", "B1", "B2", "B3"],
    ["C0", "C1", "C2", "C3"],
    ["D0", "D1", "D2", "D3"]
]
row_headers = ["CIFAR10", 'CIFAR100', 'STL10', 'Tiny ImageNet']
col_headers = [ "Error Rate", "NC1", "NC2", "NC3"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

for row, dset in zip(['A', 'B', 'C', 'D'], ['cifar10', 'cifar100', 'stl10', 'tinyi']):

    model = 'r18' if dset == 'cifar10' else 'r50'
    exp_ce = f'{dset}_{model}_ls0_s1'
    exp_ls = f'{dset}_{model}_ls0.05_s1'

    df_ce = pd.read_csv(os.path.join(path, dset, exp_ce, 'progress_s1.csv'))
    df_ls = pd.read_csv(os.path.join(path, dset, exp_ls, 'progress_s1.csv'))
    print(f'loading {dset}...')

    ticks = [0, 200, 400, 600, 800] if dset not in ['tinyi'] else [0, 100, 200, 300]

    #=============== error ===============
    i = row + '0'
    axes[i].plot(df_ce['Step'], 1 - df_ce['train_acc'], label='CE-train error')
    axes[i].plot(df_ls['Step'], 1 - df_ls['train_acc'], label='LS-train error')
    axes[i].plot(df_ce['Step'], 1 - df_ce['val_acc'], label='CE-test error', color='C0', linestyle='--')
    axes[i].plot(df_ls['Step'], 1 - df_ls['val_acc'], label='LS-test error', color='C1', linestyle='--')
    axes[i].set_ylabel('Error Rate')
    axes[i].grid(True, linestyle='--')
    if row=='D':
        axes[i].set_xlabel('Epoch')
    axes[i].set_xticks(ticks)
    if row == 'A':
        axes[i].legend()

    # =============== NC ===============
    i = row + '1'
    axes[i].plot(df_ce['Step'], df_ce['train_nc1'], label='CE')
    axes[i].plot(df_ls['Step'], df_ls['train_nc1'], label='LS')
    axes[i].set_ylabel('NC1')
    axes[i].set_yscale("log")
    axes[i].set_xticks(ticks)
    axes[i].grid(True, linestyle='--')
    if row == 'A':
        axes[i].legend()
    if row == 'D':
        axes[i].set_xlabel('Epoch')

    # =============== NC ===============
    i = row + '2'
    axes[i].plot(df_ce['Step'], df_ce['train_nc2'], label='CE', color='C0')
    axes[i].plot(df_ls['Step'], df_ls['train_nc2'], label='LS', color='C1')
    axes[i].set_ylabel('NC2')
    axes[i].set_xticks(ticks)
    axes[i].grid(True, linestyle='--')
    if row == 'A':
        axes[i].legend()
    if row == 'D':
        axes[i].set_xlabel('Epoch')

    # =============== NC ===============
    i = row + '3'
    axes[i].plot(df_ce['Step'], df_ce['train_nc3'], label='CE', color='C0')
    axes[i].plot(df_ls['Step'], df_ls['train_nc3'], label='LS', color='C1')
    axes[i].set_ylabel('NC3')
    axes[i].set_xticks(ticks)
    axes[i].grid(True, linestyle='--')
    if row == 'D':
        axes[i].set_xlabel('Epoch')
    if row == 'A':
        axes[i].legend()

plt.show()


