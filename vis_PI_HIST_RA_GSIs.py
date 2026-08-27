# Visualization for PI-HIST (R&A)'s result of graph superfamily identification (GSI) on LFR synthetic graphs

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import numpy as np
import pickle

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=64) # 64
    parser.add_argument('--P_L', type=int, default=7)
    parser.add_argument('--P_eps', type=float, default=0.5)
    parser.add_argument('--P_norm', type=str, default='no') # no, l2, z
    parser.add_argument('--P_act', type=str, default='no') # no, tanh, sig, relu, exp
    parser.add_argument('--I_L', type=int, default=7)
    parser.add_argument('--I_eps', type=float, default=0.5)
    parser.add_argument('--I_norm', type=str, default='no') # no, l2, z
    parser.add_argument('--I_act', type=str, default='no') # no, tanh, sig, relu, exp
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    # ====================
    emb_dim = args.d
    pos_RW_len = args.P_L
    pos_eps = args.P_eps
    pos_norm = args.P_norm
    pos_act = args.P_act
    ide_RW_len = args.I_L
    ide_eps = args.I_eps
    ide_norm = args.I_norm
    ide_act = args.I_act
    alpha = args.alpha
    # ==========
    num_runs = 10 # Number of independent runs
    num_graphs = 9 # Number of synthetic graphs

    # ====================
    # Load node-wise pos emb, i.e., PI-HIST (R), of all ind runs
    pkl_file = open('emb/PI-HIST(R)_LFR_L=%d_eps=%.1f_act=%s_norm=%s.pickle'
                    % (pos_RW_len, pos_eps, pos_act, pos_norm), 'rb')
    pos_embs_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    # Load node-wise ide emb, i.e., PI-HIST (A), of all ind runs
    pkl_file = open('emb/PI-HIST(A)_LFR_L=%d_eps=%.1f_act=%s_norm=%s.pickle'
                    % (ide_RW_len, ide_eps, ide_act, ide_norm), 'rb')
    ide_embs_list = pickle.load(pkl_file)
    pkl_file.close()

    # ====================
    # Average correlation matrix
    avg_cor_mat = np.zeros((num_graphs, num_graphs))
    for t in range(num_runs):
        # ====================
        # List of node-wise pos & ide emb of G_1-G_9 in current run
        pos_embs = pos_embs_list[t]
        ide_embs = ide_embs_list[t]

        # ====================
        G_embs = [] # List of graph-level emb (i.e., avg emb)
        for s in range(num_graphs):
            # ====================
            # Node-wise pos & ide emb of each syn graph
            pos_emb = pos_embs[s]
            ide_emb = ide_embs[s]

            # ====================
            # Linear combination of pos & ide emb
            emb = alpha*pos_emb + (1-alpha)*ide_emb
            # ==========
            # Optional normalization
            #_, emb_dim_ = emb.shape
            #emb = normalize(emb, 'l2')
            #for j in range(emb.shape[1]):
            #    crt_mean = np.mean(emb[:, j])
            #    crt_std = np.std(emb[:, j])
            #    if crt_std > 0.0:
            #        emb[:, j] = (emb[:, j] - crt_mean) / crt_std
            emb = normalize(emb, 'l2')
            # ==========
            del ide_emb, pos_emb

            # ====================
            # Derive the corresponding graph-level emb
            G_emb = np.mean(emb, axis=0)
            G_emb = np.reshape(G_emb, (1, -1))
            G_embs.append(G_emb)
        # ==========
        # Get correlation matrix using graph-level emb
        G_feat = np.concatenate(G_embs, axis=0)
        cor_mat = np.corrcoef(G_feat)
        # ==========
        avg_cor_mat += cor_mat
    # ==========
    avg_cor_mat = avg_cor_mat / num_runs

    # =====================
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

    plt.title(r'PI-HIST (R&A), $\alpha$=%.1f' % (alpha), fontsize=16, weight='bold', pad=15, loc='left')

    # ====================
    # Set font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # ====================
    plt.tight_layout()
    plt.show()
    #plt.savefig('vis/PI-HIST(R&A)_GSIs_a=%.1f_P-L=%d_P-eps=%.1f_P-act=%s_P-norm=%s_I-L=%d_I-eps=%.1f_I-act=%s_I-norm=%s.svg'
    #    % (alpha,
    #       pos_RW_len, pos_eps, pos_act, pos_norm,
    #       ide_RW_len, ide_eps, ide_act, ide_norm),
    #    format='svg',
    #    bbox_inches='tight',
    #    transparent=False)
    plt.close()
