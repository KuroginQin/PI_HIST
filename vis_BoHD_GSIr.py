# Visualization for Bag of high-order degree (BoHD)'s result of graph superfamily identification (GSI) on real graphs

import numpy as np
from collections import Counter

import pickle
import timeit
import matplotlib.pyplot as plt
import seaborn as sns

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
data_name_list = ['europe', 'usa', 'film', 'actor', 'dblp', 'amazon', 'blogcatalog', 'ppi']
num_graphs = len(data_name_list)
order = 5

# =====================
neighs_list = []
degs_list = []
for data_name in data_name_list:
    # ====================
    pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
    edges = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    num_edges = len(edges)
    num_nodes = np.max(edges) + 1

    # ====================
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
    neighs_list.append(neighs)
    degs_list.append(degs)

# ====================
time_s = timeit.default_timer()
hdeg_cnts_r = [[] for _ in range(order)]
for s in range(num_graphs):
    # ==========
    neighs = neighs_list[s]
    degs = degs_list[s]
    num_nodes = len(degs)
    # ==========
    hdeg_cnts = get_hdeg_cnts(neighs, degs, num_nodes, order)
    for r in range(order):
        hdeg_cnts_r[r].append(hdeg_cnts[r])
# ====================
# Merge stat to a feature matrix
feat_mat = None
for r in range(order):
    keys = sorted(set().union(*(feat.keys() for feat in hdeg_cnts_r[r])))
    feat_mat_lcl = np.zeros((num_graphs, len(keys)))
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
print('FEAT TIME %f' % (feat_time))

cor_mat = np.corrcoef(feat_mat)
min_val = np.min(cor_mat)
max_val = np.max(cor_mat[np.abs(cor_mat - 1.0) > 1e-6])

# =====================
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

plt.title('BoHD', fontsize=16, weight='bold', pad=15, loc='left')

# ====================
plt.tight_layout()
plt.show()
plt.close()
