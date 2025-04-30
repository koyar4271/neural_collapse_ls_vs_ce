import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
path = 'results_0/wandb'


fig, axes = plt.subplots(1, 4)
# ========== plot Accuracy vs. epochs
for i, dset in enumerate(['cifar10', 'cifar100', 'stl10', 'tinyi']):

    model = 'r18' if dset == 'cifar10' else 'r50'
    exp_ce = f'{dset}_{model}_ls0_s1'
    exp_ls = f'{dset}_{model}_ls0.05_s1'
    df_ce = pd.read_csv(os.path.join(path, dset, exp_ce, 'progress_s1.csv'))
    df_ls = pd.read_csv(os.path.join(path, dset, exp_ls, 'progress_s1.csv'))
    print(f'loading {dset}...')

    # ============= plot NC1 vs NC2 and Testing Acc
    if dset == 'cifar10':
        vmin, vmax = 0.10, 0.15
    elif dset == 'cifar100':
        vmin, vmax = 0.36, 1-0.60
    elif dset == 'stl10':
        vmin, vmax = 0.39, 0.45
    elif dset == 'tinyi':
        vmin, vmax = 0.4, 0.45


    p = axes[i].scatter(np.array(df_ls['train_nc2']), np.array(df_ls['train_nc1']), c=1-np.array(df_ls['val_acc']),
                   cmap= plt.cm.viridis.reversed(), vmin=vmin, vmax=vmax, label='LS', marker='^')
    axes[i].scatter(np.array(df_ce['train_nc2']), np.array(df_ce['train_nc1']), c=1-np.array(df_ce['val_acc']),
                    cmap=plt.cm.viridis.reversed(), vmin=vmin, vmax=vmax, label='CE', marker='+')
    axes[i].set_xlabel('NC2')
    axes[i].set_ylabel('NC1')
    axes[i].legend()
    if dset == 'cifar10':
        axes[i].set_xlim(0, 0.4)
        axes[i].set_ylim(5e-4, 10)
    elif dset == 'cifar100':
        axes[i].set_xlim(0.25, 1.03+0.05)
        # axes[i].set_ylim(0.28, 1e3)
    elif dset == 'stl10':
        axes[i].set_xlim(0, 1.0)
        axes[i].set_ylim(0.1, 1000)
    elif dset == 'tinyi':
        axes[i].set_xlim(0.5, 1.25)
        axes[i].set_ylim(5, 1000)

    axes[i].set_yscale("log")
    title = dset.upper() if dset != 'tinyi' else 'Tiny ImageNet'
    axes[i].set_title(title)

    fig.colorbar(p, ax=axes[i])
plt.show()