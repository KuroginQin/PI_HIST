# Visualization for evaluation results of node-level tasks on graphs w/ node identity ground-truth

import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
import argparse

# ====================
data_name_map={
    'europe': 'Europe',
    'usa': 'USA',
    'actor': 'Actor',
    'film': 'Film',
    'ppi': 'PPI',
    'blogcatalog': 'BlogCatalog',
    'dblp': 'DBLP',
    'amazon': 'Amazon'
}
# ==========
# Quality metric
qlt_lbl_map={
    'micro': 'Micro-F1',
    'macro': 'Macro-F1',
    'mod': 'Mod'
}

# ====================
# Define dot colors
# pos emb: red (#C82423)
# ide emb - green (#287885)
# unclear: - purple (#9600d4)
colors = [
    '#C82423', # node2vec
    '#C82423', # PhUSION (P)
    '#C82423', # PaCEr (P)
    '#287885', # struc2vec
    '#287885', # PhUSION (I)
    '#287885', # PaCEr (I)
    '#9600d4', # RandNE
    '#C82423', # LouvainNE
    '#C82423', # SketchNE
    '#C82423', # node2binary
    '#C82423', # S3GC
    '#C82423', # MAGI
    '#287885', # GraLSP
    '#287885', # DRSR
    '#9600d4', # DGI
    '#9600d4', # GraphMAE2
    '#9600d4', # GGD
    '#9600d4', # E2Neg
    '#F8AC8C', # PI-HIST (R)
    '#9AC9DB'  # PI-HIST (A)
]
# ==========
# Define dot shapes
markers = [
    's', # node2vec
    'd', # PhUSION (P)
    'h', # PaCEr (P)
    's', # struc2vec
    'd', # PhUSION (I)
    'h', # PaCEr (I)
    '>', # RandNE
    'v', # LouvainNE
    '^', # SketchNE
    '<', # node2binary
    'X', # S3GC
    'P', # MAGI
    'X', # GraphLSP
    'P', # DRSR
    'D', # DGI
    'p', # GraphMAE2
    's', # DDG
    '8', # E2Neg
    'o', # PI-HIST (P)
    'o'  # PI-HIST (I)
]
# ==========
# Define dot edge colors
# efficient/scalable emb: black
edge_clr = [
    'none', # node2vec
    'none', # PhUSION (P)
    'none', # PaCEr (P)
    'none', # struc2vec
    'none', # PhUSION (I)
    'none', # PaCEr (I)
    'black', # RandNE
    'black', # LouvainNE
    'black', # SketchNE
    'black', # node2binary
    'black', # S3GC
    'black', # MAGI
    'none', # GraLSP
    'none', # DRSR
    'black', # DGI
    'black', # GraphMAE2
    'black', # GGD
    'black', # E2Neg
    '#C82423', # PI-HIST (R)
    '#287885' # PI-HIST (A)
]

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--qlt_lbl', type=str) # macro, micro, mod
    parser.add_argument('--log_flag', type=bool, default=True)
    args = parser.parse_args()

    # ====================
    data_name = args.data_name
    qlt_lbl = args.qlt_lbl
    log_flag = args.log_flag

    # ====================
    # Load evaluation results
    df = pd.read_csv('res/nc_%s_res.csv' % (data_name))
    num_meth = len(df)

    # ====================
    # Get quality & efficincy rankings of PI-HIST (R) & PI-HIST (A)
    Qs = df[qlt_lbl_map[qlt_lbl]]
    Es = df['Time']
    # ==========
    if qlt_lbl == 'cond':
        df['Q_Rank'] = Qs.rank(method='min', ascending=True)
    else:
        df['Q_Rank'] = Qs.rank(method='min', ascending=False)
    P_Q_rank = df['Q_Rank'][num_meth - 2]
    I_Q_rank = df['Q_Rank'][num_meth - 1]
    # ==========
    df['E_Rank'] = Es.rank(method='min', ascending=True)
    P_E_rank = df['E_Rank'][num_meth - 2]
    I_E_rank = df['E_Rank'][num_meth - 1]

    # ====================
    # Pre-set opacity
    alphas = [0.6 for _ in range(num_meth)]
    alphas[-2] = 1.0
    alphas[-1] = 1.0

    # ====================
    fig, ax = plt.subplots(figsize=(5.5, 6))
    txts = []
    for i, (_, row) in enumerate(df.iterrows()):
        # ====================
        method = row['Methods']
        if method == 'PI-HIST (R)':
            method = r'$\mathbf{PI}$-$\mathbf{HIST}$ (R)'
        elif method == 'PI-HIST (A)':
            method = r'$\mathbf{PI}$-$\mathbf{HIST}$ (A)'
        # ====================
        if log_flag:
            time = np.log10(row['Time']) # np.log10, np.log, np.log2
        else:
            time = row['Time']
        qlt = row[qlt_lbl_map[qlt_lbl]]

        # ====================
        # Draw scatter
        ax.scatter(
            time,
            qlt,
            color=colors[i],
            marker=markers[i],
            edgecolors=edge_clr[i],
            s=120,
            alpha=alphas[i],
            linewidth=1.5,
            label=method
        )
        # ==========
        txt = plt.text(time, qlt, '', fontsize=0, fontweight='bold', color='#DC0000')
        txts.append(txt)
    # ====================
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1
    )

    # ====================
    r_q = df[qlt_lbl_map[qlt_lbl]][num_meth-2]
    r_e = df['Time'][num_meth-2]
    if log_flag:
        r_e = np.log10(r_e)
    txt_1 = plt.text(r_e, r_q, '(Q:%d,E:%d)' % (P_Q_rank, P_E_rank), fontsize=12, fontweight='bold', color='#DC0000')
    txts[-2] = txt_1
    # ==========
    a_q = df[qlt_lbl_map[qlt_lbl]][num_meth-1]
    a_e = df['Time'][num_meth-1]
    if log_flag:
        a_e = np.log10(a_e)
    txt_2 = plt.text(a_e, a_q, '(Q:%d,E:%d)' % (I_Q_rank, I_E_rank), fontsize=12, fontweight='bold', color='#DC0000')
    txts[-1] = txt_2
    # ==========
    adjust_text(
        txts,
        force_static=5.0,
        force_text=5.0,
        expand_text=(5, 5),
        lim=5000,
        precision=0.01,
        #nudge_x=1.0, nudge_y=1.0 # Only for europe & mod
    )

    # ====================
    # Set figure title
    if qlt_lbl == 'micro' or qlt_lbl == 'macro':
        plt.title('Node Identity Classification \non ' + data_name_map[data_name], fontsize=14, fontweight='bold')
    elif qlt_lbl == 'mod':
        plt.title('Community Detection \non ' + data_name_map[data_name], fontsize=14, fontweight='bold')

    # ====================
    # Set labels of x & y axies
    x_label = r'log [Time (s)]$\downarrow$' if log_flag else r'Time (s)$\downarrow$'
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold', labelpad=10)
    # ==========
    if qlt_lbl == 'micro':
        ax.set_ylabel(r'Micro-F1 (%)$\uparrow$', fontsize=14, fontweight='bold', labelpad=10)
    elif qlt_lbl == 'macro':
        ax.set_ylabel(r'Macro-F1 (%)$\uparrow$', fontsize=14, fontweight='bold', labelpad=10)
    elif qlt_lbl == 'mod':
        ax.set_ylabel(r'Modularity (%)$\uparrow$', fontsize=14, fontweight='bold', labelpad=10)

    # ====================
    # Add grid
    ax.grid(True, alpha=0.3, linewidth=1.0)

    # ====================
    # Set font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    plt.tight_layout()
    plt.show()
    #plt.savefig('vis/nig_%s_%s.svg' % (data_name, qlt_lbl),
    #            format='svg',
    #            bbox_inches='tight'
    #            )
    plt.close()
