

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Plot.utils import add_headers
import matplotlib
matplotlib.use('TkAgg')
path = 'results_0/wandb'


# ========= color mapping =========
# Normalize x values to [0, 1] for colormap mapping
cmap1 = plt.cm.viridis
cmap2 = plt.cm.plasma
norm = plt.Normalize(1, 7)


# ============================ Plot ============================

mosaic = [
    ["A0", "A1", "A2", "A3"],
    ["B0", "B1", "B2", "B3"],
]
row_headers = ["Train Loss", "Log-Scaled Train Loss"]
col_headers = ["CIFAR10", "CIFAR100", 'STL10', "Tiny ImageNet"]


subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

for col, dset in zip(['0', '1', '2', '3'], ['cifar10', 'cifar100', 'stl10', 'tinyi']):
    print(f'col={col}, dset={dset}')

    model = 'r18' if dset == 'cifar10' else 'r50'
    exp_ce = f'{dset}_{model}_ls0_s1'
    exp_ls = f'{dset}_{model}_ls0.05_s1'

    filename = 'progress_s1.csv'
    df_ce = pd.read_csv(os.path.join(path, dset, exp_ce, filename))
    df_ls = pd.read_csv(os.path.join(path, dset, exp_ls, filename))
    print(f'loading {dset} from {exp_ce} and {exp_ls}')

    if dset in ['cifar10']:
        eps = 0.002
    elif dset in ['cifar100']:
        eps = 0.006
    elif dset in [ 'tinyi']:
        eps = 0.005
    else:
        eps = 0.01

    loss_min_ce = np.min(df_ce['loss'].values)
    loss_max_ce = np.max(df_ce['loss'].values)
    loss_min_ls = np.min(df_ls['loss'].values)
    loss_max_ls = np.max(df_ls['loss'].values)



    # Select loss values at specific epochs
    epochs = [9, 19, 49, 99]
    log_loss_ce = np.log(df_ce.loc[df_ce['Step'].isin(epochs), 'loss'].values - loss_min_ce + eps)
    log_loss_ls = np.log(df_ls.loc[df_ls['Step'].isin(epochs), 'loss'].values - loss_min_ls + eps)

    # Print results
    for epoch, loss_ce, loss_ls in zip(epochs, log_loss_ce, log_loss_ls):
        print(f"Epoch {epoch}: log_loss_ce = {loss_ce:.4f}, log_loss_ls = {loss_ls:.4f}")






    if dset in ['cifar10', 'cifar100', 'stl10']:
        num_epochs = 400
    else:
        num_epochs = 150
    df_ce = df_ce[df_ce['Step'] < num_epochs]
    df_ls = df_ls[df_ls['Step'] < num_epochs]

    # ==== Training Acc
    ax = axes[f'A{col}']
    ax.plot(df_ce['Step'], (df_ce['loss'].values), label='$\delta=0$', color='C0')
    ax.plot(df_ls['Step'], (df_ls['loss'].values), label='$\delta=0.05$', color='C1')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, linestyle='--')

    # ==== Training Acc
    ax = axes[f'B{col}']
    ax.plot(df_ce['Step'], np.log(df_ce['loss'].values - loss_min_ce + eps), label='$\delta=0$', color='C0')
    ax.plot(df_ls['Step'], np.log(df_ls['loss'].values - loss_min_ls + eps), label='$\delta=0.05$', color='C1')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, linestyle='--')

    print(f'{dset} min loss ce {loss_min_ce} ls {loss_min_ls}')
plt.show()




