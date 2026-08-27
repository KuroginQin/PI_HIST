# Visualization for evaluation results of node-level tasks on large graphs w/ node position ground-truth

import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
import argparse

# ====================
data_name_map = {
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
# Map from method names to their abbreviations
meth_name_map={
    'node2vec': 'n2v',
	'PhUSION (P)': 'PhN(P)',
	'PaCEr (P)': 'PaCEr(P)',
	'struc2vec': 's2v',
    'PhUSION (I)': 'PhN(I)',
	'PaCEr (I)': 'PaCEr(I)',
	'RandNE': 'RandNE',
	'LouvainNE': 'LvnNE',
	'SketchNE': 'SktNE',
	'node2binary': 'n2b',
	'S3GC': 'S3GC',
	'MAGI': 'MAGI',
	'GraLSP': 'GLSP',
	'DRSR': 'DRSR',
	'DGI': 'DGI',
	'GraphMAE2': 'GMAE2',
	'GGD': 'GGD',
	'E2Neg': 'E2Neg',
	'PI-HIST (R)': 'PI-HIST (R)',
	'PI-HIST (A)': 'PI-HIST (A)'
}
# ==========
# Quality metric
qlt_lbl_map = {
    'micro': 'Micro-F1',
    'macro': 'Macro-F1',
    'cond': 'Cond'
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
    's',  # node2vec
    'd',  # PhUSION (P)
    'h',  # PaCEr (P)
    's',  # struc2vec
    'd',  # PhUSION (I)
    'h',  # PaCEr (I)
    '>',  # RandNE
    'v',  # LouvainNE
    '^',  # SketchNE
    '<',  # node2binary
    'X',  # S3GC
    'P',  # MAGI
    'X',  # GraphLSP
    'P',  # DRSR
    'D',  # DGI
    'p',  # GraphMAE2
    's',  # DDG
    '8',  # E2Neg
    'o',  # PI-HIST (P)
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
    # Identify OOM/OOT (i.e., invalid) methods
    df['status'] = 'valid'
    for idx, row in df.iterrows():
        if row['Time'] == 'OOM':
            df.at[idx, 'status'] = 'OOM'
        elif row['Time'] == 'OOT':
            df.at[idx, 'status'] = 'OOT'
    # ==========
    val_df = df[df['status'] == 'valid'].copy()
    inval_df = df[df['status'].isin(['OOM', 'OOT'])].copy()

    # ====================
    # Get quality & efficincy rankings of PI-HIST (R) & PI-HIST (A)
    P_Q_rank = P_E_rank = I_Q_rank = I_E_rank = -1
    if len(val_df) > 0:
        # ====================
        val_df['Time_numeric'] = pd.to_numeric(val_df['Time'])
        val_df['Q_numeric'] = pd.to_numeric(val_df[qlt_lbl_map[qlt_lbl]])
        # ==========
        if qlt_lbl == 'cond':
            val_df['Q_Rank'] = val_df['Q_numeric'].rank(method='min', ascending=True)
        else:
            val_df['Q_Rank'] = val_df['Q_numeric'].rank(method='min', ascending=False)
        # ==========
        val_df['E_Rank'] = val_df['Time_numeric'].rank(method='min', ascending=True)

        # ====================
        R_row = val_df[val_df['Methods'] == 'PI-HIST (R)']
        A_row = val_df[val_df['Methods'] == 'PI-HIST (A)']

        if not R_row.empty:
            P_Q_rank = R_row['Q_Rank'].values[0]
            P_E_rank = R_row['E_Rank'].values[0]
        if not A_row.empty:
            I_Q_rank = A_row['Q_Rank'].values[0]
            I_E_rank = A_row['E_Rank'].values[0]

    # ====================
    # Pre-set opacity
    alphas = [0.6 for _ in range(num_meth)]
    R_idx = df[df['Methods'] == 'PI-HIST (R)'].index[0] if 'PI-HIST (R)' in df['Methods'].values else -1
    A_idx = df[df['Methods'] == 'PI-HIST (A)'].index[0] if 'PI-HIST (A)' in df['Methods'].values else -1
    if R_idx != -1:
        alphas[R_idx] = 1.0
    if A_idx != -1:
        alphas[A_idx] = 1.0

    # ====================
    fig, ax = plt.subplots(figsize=(5.5, 6))
    # ==========
    # Select OOM/OOT (i.e., invalid) methods
    inval_idxs = inval_df.index.tolist()
    for i in inval_idxs:
        # ====================
        row = df.iloc[i]
        meth_name = row['Methods']
        status = row['status']
        inval_label = f"{meth_name_map[meth_name]}:{status}"
        # ==========
        # Draw scatter for OOM/OOT methods
        ax.scatter(
            [], [],
            color=colors[i],
            marker=markers[i],
            edgecolors=edge_clr[i],
            s=120,
            alpha=0.5,
            linewidth=1.5,
            label=inval_label,
            visible=False
        )
    # ==========
    # Select valid methods
    val_idxs = val_df.index.tolist()
    txts = []
    for i in val_idxs:
        # ====================
        row = df.iloc[i]
        meth_name = row['Methods']
        # ==========
        if meth_name == 'PI-HIST (R)':
            meth_label = r'$\mathbf{PI}$-$\mathbf{HIST}$ (R)'
        elif meth_name == 'PI-HIST (A)':
            meth_label = r'$\mathbf{PI}$-$\mathbf{HIST}$ (A)'
        else:
            meth_label = meth_name_map[meth_name]
        # ==========
        time_val = pd.to_numeric(row['Time'])
        qlt_val = pd.to_numeric(row[qlt_lbl_map[qlt_lbl]])
        if qlt_lbl == 'cond':
            qlt_val = 100 - qlt_val

        # ====================
        if log_flag:
            time_plot = np.log10(time_val)
        else:
            time_plot = time_val

        # ====================
        # Draw scatter for valid methods
        ax.scatter(
            time_plot,
            qlt_val,
            color=colors[i],
            marker=markers[i],
            edgecolors=edge_clr[i],
            s=120,
            alpha=alphas[i],
            linewidth=1.5,
            label=meth_label
        )
        # ==========
        txt = plt.text(time_val, qlt_val, '', fontsize=0, fontweight='bold', color='#DC0000')
        txts.append(txt)

    # ====================
    # Set legend
    handles, labels = ax.get_legend_handles_labels()
    # ==========
    val_mask = [not ('[OOM]' in label or '[OOT]' in label) for label in labels]
    # ==========
    val_handles = [h for i, h in enumerate(handles) if val_mask[i]]
    val_labels = [l for i, l in enumerate(labels) if val_mask[i]]
    inval_handles = [h for i, h in enumerate(handles) if not val_mask[i]]
    inval_labels = [l for i, l in enumerate(labels) if not val_mask[i]]

    # ====================
    new_handles = []
    new_labels = []
    # ==========
    # OOM/OOT methods
    if inval_handles:
        new_handles.extend(inval_handles)
        new_labels.extend(inval_labels)
    # ==========
    # Valid methods
    if val_handles:
        new_handles.extend(val_handles)
        new_labels.extend(val_labels)
    # ==========
    ax.legend(
        new_handles,
        new_labels,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=11.5,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1
    )

    # ====================
    r_q = pd.to_numeric(val_df[qlt_lbl_map[qlt_lbl]][num_meth-2])
    if qlt_lbl == 'cond':
        r_q = 100 - r_q
    r_e = pd.to_numeric(val_df['Time'][num_meth-2])
    if log_flag:
        r_e = np.log10(r_e)
    txt_1 = plt.text(r_e, r_q, '(Q:%d,E:%d)' % (P_Q_rank, P_E_rank), fontsize=12, fontweight='bold', color='#DC0000')
    txts[-2] = txt_1
    # ==========
    a_q = pd.to_numeric(df[qlt_lbl_map[qlt_lbl]][num_meth-1])
    if qlt_lbl == 'cond':
        a_q = 100 - a_q
    a_e = pd.to_numeric(df['Time'][num_meth-1])
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
    )

    # ====================
    # Set figure title
    if qlt_lbl in ['micro', 'macro']:
        plt.title(f'Node Position Classification \non {data_name_map[data_name]}', fontsize=14, fontweight='bold')
    else:
        plt.title(f'Node Identity Clustering \non {data_name_map[data_name]}', fontsize=14, fontweight='bold')

    # ====================
    # Set labels of x & y axies
    x_label = r'log [Time (s)]$\downarrow$' if log_flag else r'Time (s)$\downarrow$'
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold', labelpad=10)
    # ==========
    if qlt_lbl == 'micro':
        ax.set_ylabel(r'Micro-F1 (%)$\uparrow$', fontsize=14, fontweight='bold', labelpad=10)
    elif qlt_lbl == 'macro':
        ax.set_ylabel(r'Macro-F1 (%)$\uparrow$', fontsize=14, fontweight='bold', labelpad=10)
    elif qlt_lbl == 'cond':
        #ax.set_ylabel(r'Conductance (%)$\downarrow$', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel(r'[ 1 - Conductance$\downarrow$] (%)', fontsize=14, fontweight='bold', labelpad=10)

    # ====================
    # Add grid
    ax.grid(True, alpha=0.3, linewidth=1.0)

    # ====================
    # Set font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    plt.tight_layout()
    plt.show()
    #plt.savefig('vis/npg_%s_%s.svg' % (data_name, qlt_lbl),
    #            format='svg',
    #            bbox_inches='tight'
    #            )
    plt.close()
