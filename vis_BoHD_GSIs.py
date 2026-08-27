# Visualization for Bag of high-order degree (BoHD)'s result of graph superfamily identification (GSI) on synthetic graphs

import numpy as np
from collections import Counter

import pickle
import timeit
import matplotlib.pyplot as plt
import seaborn as sns

def get_hdeg_cnts(neighs_1st, degs, num_nodes, order=5):
    # ====================
    h_deg_cnts = []
    # ==========
    cnt = Counter()
    for d in degs:
        cnt[d] += 1
    h_deg_cnts.append(cnt)
    del cnt
    # ==========
    pre_neighs = neighs_1st.copy()
    for r in range(1, order):
        # ==========
        neighs = [] # Neighbors of each node w.r.t. the r-th order
        for i in range(num_nodes):
            cur_neigh = []
            for j in pre_neighs[i]:
                for n in neighs_1st[j]:
                    cur_neigh.append(n)
            neighs.append(list(set(cur_neigh)))
        # ==========
        cnt = Counter()
        for i in range(num_nodes):
            for j in neighs[i]:
                cnt[degs[j]] += 1
        h_deg_cnts.append(cnt)
        del cnt

    return h_deg_cnts

# ====================
num_runs = 10  # Number of independent runs
num_graphs = 9  # Number of syn graph (in each run)
num_nodes = 1000
# ==========
c_min = 100
c_min_list = [20, 40, 60, 80, 100]
c_max = 500
# ==========
k_avg = 10
k_avg_list = [10, 9, 8, 7, 6]
k_max = 50
# ==========
mu = 0.1

order = 5 # Order of neighbors considered

# =====================
neighs_list_gbl = [[] for _ in range(num_runs)] # List of 1st-order neighbors for all nodes in all runs
degs_list_gbl = [[] for _ in range(num_runs)] # List of node deg for all graphs in all runs
# =====================
for c_min_ in c_min_list:
    # ====================
    pkl_file = open('data/LFR_edges_list_n=%d_mu=%.1f_k=%d_maxk=%d_minc=%d_maxc=%d.pickle'
                    % (num_nodes, mu, k_avg, k_max, c_min_, c_max), 'rb')
    edges_list = pickle.load(pkl_file)
    pkl_file.close()
    # ====================
    for t in range(num_runs):
        # ==========
        edges = edges_list[t]
        neighs = [[] for _ in range(num_nodes)]
        degs = [0 for _ in range(num_nodes)]
        # ==========
        for (src, dst) in edges:
            # ==========
            neighs[src].append(dst)
            neighs[dst].append(src)
            # ==========
            degs[src] += 1
            degs[dst] += 1
        # ==========
        neighs_list_gbl[t].append(neighs)
        degs_list_gbl[t].append(degs)
# =====================
for k_avg_ in k_avg_list:
    if k_avg_ == k_avg_list[0]: continue  # Skip the repeated graph
    # ====================
    pkl_file = open('data/LFR_edges_list_n=%d_mu=%.1f_k=%d_maxk=%d_minc=%d_maxc=%d.pickle'
                    % (num_nodes, mu, k_avg_, k_max, c_min, c_max), 'rb')
    edges_list = pickle.load(pkl_file)
    pkl_file.close()
    # ====================
    for t in range(num_runs):
        # ==========
        edges = edges_list[t]
        # ==========
        neighs = [[] for _ in range(num_nodes)]
        degs = [0 for _ in range(num_nodes)]
        # ==========
        for (src, dst) in edges:
            # ==========
            neighs[src].append(dst)
            neighs[dst].append(src)
            # ==========
            degs[src] += 1
            degs[dst] += 1
        # ==========
        neighs_list_gbl[t].append(neighs)
        degs_list_gbl[t].append(degs)

# ====================
# Average correlation matrix
avg_cor_mat = np.zeros((num_graphs, num_graphs))
for t in range(num_runs):
    # ====================
    neighs_list = neighs_list_gbl[t]
    degs_list = degs_list_gbl[t]
    # ====================
    # Get BoHD features of current run
    time_s = timeit.default_timer()
    num_syn_set = len(neighs_list)
    hdeg_cnts_r = [[] for _ in range(order)]
    for s in range(num_syn_set):
        # ==========
        neighs = neighs_list[s]
        degs = degs_list[s]
        # ==========
        hdeg_cnts = get_hdeg_cnts(neighs, degs, num_nodes, order)
        for r in range(order):
            hdeg_cnts_r[r].append(hdeg_cnts[r])
    # ====================
    # Merge stat to a feature matrix
    feat_mat = None
    for r in range(order):
        keys = sorted(set().union(*(feat.keys() for feat in hdeg_cnts_r[r])))
        feat_mat_lcl = np.zeros((num_syn_set, len(keys)))
        # ==========
        for i, feat in enumerate(hdeg_cnts_r[r]):
            for j, key in enumerate(keys):
                feat_mat_lcl[i, j] = feat.get(key, 0)
        if r == 0:
            feat_mat = feat_mat_lcl
        else:
            feat_mat = np.concatenate((feat_mat, feat_mat_lcl), axis=1)
    # ====================
    time_e = timeit.default_timer()
    feat_time = time_e - time_s
    print('RUN-%d FEAT TIME %f' % (t+1, feat_time))

    # ====================
    # Get correlation matrix of current run
    cor_mat = np.corrcoef(feat_mat)
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

plt.title('BoHD', fontsize=16, weight='bold', pad=15, loc='left')

# ====================
# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# ====================
plt.tight_layout()
plt.show()
#plt.savefig('vis/BoHD_GSIs.svg',
#            format='svg',
#            bbox_inches='tight',
#            transparent=False)
plt.close()
