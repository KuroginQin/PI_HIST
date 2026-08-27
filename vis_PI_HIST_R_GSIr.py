# Visualization for PI-HIST (R)'s result of graph superfamily identification (GSI) on real graphs

import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import numpy as np
import pickle

# ====================
data_name_map_r={
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
data_name_map_c={
    'europe': 'Europe',
    'usa': 'USA',
    'actor': 'Actor',
    'film': 'Film',
    'ppi': 'PPI',
    'blogcatalog': 'BlogC',
    'dblp': 'DBLP',
    'amazon': 'Amazon'
}

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=64) # 64
    parser.add_argument('--L', type=int, default=7)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--norm', type=str, default='no') # no, l2, z
    parser.add_argument('--act', type=str, default='no') # no, tanh, sig, relu, exp
    args = parser.parse_args()

    # ====================
    emb_dim = args.d
    RW_len = args.L
    eps = args.eps
    norm = args.norm
    act = args.act
    # ==========
    # List of data names with layout order same at in Fig. 4a
    data_name_list = ['europe', 'usa', 'film', 'actor', 'dblp', 'amazon', 'blogcatalog', 'ppi']
    num_graphs = len(data_name_list)

    # ====================
    G_embs = []  # List of graph-level emb (i.e., avg emb)
    for t in range(num_graphs):
        # ====================
        # Load save node-wise emb of each graph
        data_name = data_name_list[t]
        pkl_file = open('emb/PI-HIST(R)_%s_L=%d_eps=%.1f_act=%s_norm=%s.pickle'
                        % (data_name, RW_len, eps, act, norm), 'rb')
        pos_emb = pickle.load(pkl_file)
        pkl_file.close()

        # ====================
        # Derive the corresponding graph-level emb
        G_emb = np.mean(pos_emb, axis=0)
        G_emb = np.reshape(G_emb, (1, -1))
        G_embs.append(G_emb)
    # ====================
    # Get correlation matrix using graph-level emb
    G_feat = np.concatenate(G_embs, axis=0)
    cor_mat = np.corrcoef(G_feat)

    # ====================
    # Visualize correlation matrix
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cor_mat,
                     cmap='RdBu_r',
                     annot=True, annot_kws={"size": 16},
                     square=True,
                     cbar_kws={'shrink': 1.0},
                     vmin=0.0, vmax=1.0,
                     linewidth=.5,
                     fmt='.2f',
                     xticklabels=[data_name_map_c[data_name] for data_name in data_name_list],
                     yticklabels=[data_name_map_r[data_name] for data_name in data_name_list]
                     )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=0)

    plt.title('PI-HIST (R)', fontsize=16, weight='bold', pad=15, loc='left')

    # ====================
    # Set font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # ====================
    plt.tight_layout()
    plt.show()
    #plt.savefig('vis/PI-HIST(R)_GSIr_L=%d_eps=%.1f_act=%s_norm=%s.svg' % (RW_len, eps, act, norm),
    #            format='svg',
    #            bbox_inches='tight',
    #            transparent=False)
    plt.close()
