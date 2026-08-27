# Visualization for PI-HIST (A)'s result of graph superfamily identification (GSI) on LFR synthetic graphs

import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import numpy as np
import pickle

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
    num_runs = 10 # Number of independent runs
    num_graphs = 9 # Number of synthetic graphs

    # ====================
    pkl_file = open('emb/PI-HIST(A)_LFR_L=%d_eps=%.1f_act=%s_norm=%s.pickle'
                    % (RW_len, eps, act, norm), 'rb')
    embs_list = pickle.load(pkl_file)
    pkl_file.close()

    # ====================
    # Average correlation matrix
    avg_cor_mat = np.zeros((num_graphs, num_graphs))
    for t in range(num_runs):
        # ====================
        # List of node-wise emb of G_1-G_9 in current run
        embs = embs_list[t]

        # ====================
        G_embs = [] # List of graph-level emb (i.e., avg emb)
        for s in range(num_graphs):
            # ====================
            # Node-wise emb of each syn graph
            emb = embs[s]

            # ====================
            # Derive the corresponding graph-level emb
            G_emb = np.mean(emb, axis=0)
            G_emb = np.reshape(G_emb, (1, -1))
            G_embs.append(G_emb)
        # ====================
        # Get correlation matrix of current run using graph-level emb
        G_feat = np.concatenate(G_embs, axis=0)
        cor_mat = np.corrcoef(G_feat)
        # ==========
        avg_cor_mat += cor_mat
    # ==========
    avg_cor_mat = avg_cor_mat / num_runs

    # =====================
    # Visualize avg correlation matrix
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(avg_cor_mat,
                     cmap='RdBu_r',
                     annot=True, annot_kws={"size": 16},
                     square=True,
                     cbar_kws={'shrink': 1.0},
                     vmin=0.0, vmax=1.0,
                     fmt='.2f',
                     linewidth=.5,
                     xticklabels=[r'$G_{%d}$' % (i) for i in range(1, num_graphs+1)],
                     yticklabels=[r'$G_{%d}$' % (i) for i in range(1, num_graphs+1)]
                     )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)

    plt.title('PI-HIST (R)', fontsize=16, weight='bold', pad=15, loc='left')

    # ====================
    plt.tight_layout()
    plt.show()
    #plt.savefig('vis/PI-HIST(A)_GSIs_L=%d_eps=%.1f_act=%s_norm=%s.svg' % (RW_len, eps, act, norm),
    #            format='svg',
    #            bbox_inches='tight',
    #            transparent=False)
    plt.close()
