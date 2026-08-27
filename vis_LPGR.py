# Visualization for evaluation results of link prediction (LP) & graph reconstruction (GR)

import pandas as pd
import matplotlib.pyplot as plt
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
meth_name_map = {
    'node2vec': 'node2vec',
    'PhUSION (P)': 'PhUSION(P)',
    'PaCEr (P)': 'PaCEr(P)',
    'struc2vec': 'struc2vec',
    'PhUSION (I)': 'PhUSION(I)',
    'PaCEr (I)': 'PaCEr(I)',
    'RandNE': 'RandNE',
    'LouvainNE': 'LouvainNE',
    'SketchNE': 'SketchNE',
    'node2binary': 'node2binary',
    'S3GC': 'S3GC',
    'MAGI': 'MAGI',
    'GraLSP': 'GraLSP',
    'DRSR': 'DRSR',
    'DGI': 'DGI',
    'GraphMAE2': 'GraphMAE2',
    'GGD': 'GGD',
    'E2Neg': 'E2Neg',
    'PI-HIST (R)': r'$\mathbf{PI}$-$\mathbf{HIST}$(R)',
    'PI-HIST (A)': r'$\mathbf{PI}$-$\mathbf{HIST}$(A)',
    'PI-HIST (R&A)': r'$\mathbf{PI}$-$\mathbf{HIST}$(R&A)'
}

# ====================
# Bar colors
# pos emb: light red (#FFC1A6)
# ide emb: light green (#D0E2C0)
# unclear: light purple (#E8D8FC)
# PI-HIST (R): red
# PI-HIST (A): green
# PI-HIST (R&A): blue
bar_clrs = [
    '#FFC1A6',  # node2vec
    '#FFC1A6',  # PhUSION (P)
    '#FFC1A6',  # PaCEr (P)
    '#D0E2C0',  # struc2vec
    '#D0E2C0',  # PhUSION (I)
    '#D0E2C0',  # PaCEr (I)
    '#E8D8FC',  # RandNE
    '#FFC1A6',  # LouvainNE
    '#FFC1A6',  # SketchNE
    '#FFC1A6',  # node2binary
    '#FFC1A6',  # S3GC
    '#FFC1A6',  # MAGI
    '#D0E2C0',  # GraLSP
    '#D0E2C0',  # DRSR
    '#E8D8FC',  # DGI
    '#E8D8FC',  # GraphMAE2
    '#E8D8FC',  # GGD
    '#E8D8FC',  # E2Neg
    '#E95351',  # PI-HIST (R)
    '#67A583',  # PI-HIST (A)
    '#5292F7'  # PI-HIST (R&A)
]
# ==========
# Method name colors
# pos emb: red (#C82423)
# ide emb - green (#287885)
# unclear: - purple (#9600d4)
# PI-HIST (R): red
# PI-HIST (A): green
# PI-HIST (R&A): blue
meth_lbl_clrs = [
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
    '#E95351', # PI-HIST (R) red
    '#67A583',  # PI-HIST (A) green
    '#5292F7' # PI-HIST (R&A) blue
]

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--task', type=str) # gr, lp
    args = parser.parse_args()

    # ====================
    data_name = args.data_name
    task = args.task

    # =====================
    # Load evaluation results
    res_df = pd.read_csv('res/%s_%s_res.csv' % (task, data_name))
    res_df.rename(columns={'Unnamed: 0': 'method'}, inplace=True)
    meths = res_df['method'].tolist()
    meth_name_list = [meth_name_map[meth] for meth in meths]
    qlt_res_list = res_df['AUC'].tolist()
    qlt_r = qlt_res_list[-3] # Result of PI-HIST (R)
    qlt_a = qlt_res_list[-2] # Result of PI-HIST (A)
    qlt_ra = qlt_res_list[-1] # Result of PI-HIST (R&A)
    # ==========
    imp_r = (qlt_ra - qlt_r) / qlt_r # Improvement of PI-HIST (R&A) w.r.t. PI-HIST (R)
    imp_a = (qlt_ra - qlt_a) / qlt_a # Improvement of PI-HIST (R&A) w.r.t. PI-HIST (A)

    # =====================
    # Load detailed results for PI-HIST (R&A)
    df_cmb = pd.read_csv('res/%s_%s_cmb_res.csv' % (task, data_name))
    alpha_cols = df_cmb.columns.tolist()[::-1]
    alphas = []
    qlt_cmb = []
    for col in alpha_cols:
        alpha_val = float(col.replace('alpha=', '')) if 'alpha=' in col else float(col)
        alphas.append(alpha_val)
        qlt_cmb.append(df_cmb.iloc[0][col])

    # ====================
    # Combine histogram & line chart
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax2 = ax1.twinx() # Shared axis of histogram & line chart

    # ====================
    # Draw histogram for all methods
    meth_y = np.arange(len(meth_name_list))
    bars = ax1.barh(meth_y, qlt_res_list, height=0.8, edgecolor='white', linewidth=0.1)
    for bar, color in zip(bars, bar_clrs):
        bar.set_facecolor(color)
        bar.set_zorder(2)
    # ==========
    ax1.set_ylim(len(meth_name_list) - 0.5, -0.5)
    ax2.set_ylim(len(meth_name_list) + 0.0, -1.0)

    # ====================
    # Draw reference lines of PI-HIST (R), PI-HIST (A), & PI-HIST (R&A)
    cmb_y = np.linspace(len(meth_name_list) - 0.5, -0.5, len(alphas))
    ax2.plot([qlt_r for _ in range(len(alphas))], cmb_y, color='#E95351', linestyle=':', linewidth=2)
    ax2.plot([qlt_a for _ in range(len(alphas))], cmb_y, color='#67A583', linestyle=':', linewidth=2)
    ax2.plot([qlt_ra for _ in range(len(alphas))], cmb_y, color='#5292F7', linestyle=':', linewidth=2)
    # ====================
    # Draw line chart of PI-HIST (R&A)
    ax2.plot(qlt_cmb, cmb_y, color='#1E3A8A', linewidth=3, marker='o',
             markersize=6, markerfacecolor='#1E3A8A', markeredgewidth=0, label='PI-HIST(R&A)')
    # ==========
    # Set legend for line chart of PI-HIST (R&A)
    ax2.legend(
        loc='upper center',
        bbox_to_anchor=(0.95, 1.1), # Adjust position of legend box
        fontsize=12,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        handletextpad=0.5
    )

    # ====================
    # Set figure title
    if task == 'lp':
        plt.title('Link Prediction on ' + data_name_map[data_name], fontsize=14, fontweight='bold')
    elif task == 'gr':
        plt.title('Graph Reconstruction on ' + data_name_map[data_name], fontsize=14, fontweight='bold')

    # ====================
    # Set method names (shard y-axis)
    ax1.set_yticks(meth_y)
    ax1.set_yticklabels(meth_name_list, ha='right', fontsize=12)
    for lbl, color in zip(ax1.get_yticklabels(), meth_lbl_clrs):
        lbl.set_color(color)
    # ==========
    # Set alpha values/labels (shared y-axis)
    ax2.set_yticks(cmb_y)
    alpha_lbl = []
    for a in alphas:
        if a == 0.0:
            alpha_lbl.append(r'$\alpha$=%.1f' % (a))
        else:
            alpha_lbl.append(f'{a:.1f}')
    ax2.set_yticklabels(alpha_lbl, fontsize=12)

    # ====================
    # Set x-axis w/ quality metric & improvements
    ax1.set_xlabel(r'AUC (%) $\uparrow$' + ' (R:+%.2f%%, A:+%.2f%%)' % (imp_r*100, imp_a*100),
                   fontsize=14, fontweight='bold')
    ax1.set_xlim(min(qlt_res_list)-0.2, max(qlt_res_list)+0.1)

    # ====================
    # Set font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # ====================
    plt.tight_layout()
    plt.show()
    #plt.savefig('vis/%s_%s.svg' % (task, data_name),
    #            format='svg',
    #            bbox_inches='tight',
    #            transparent=False)
    plt.close()