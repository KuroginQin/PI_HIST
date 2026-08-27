# Visualization for subgraph significance profile (SSP)'s result of graph superfamily identification (GSI) on real graphs

import numpy as np
import pickle

from sklearn.preprocessing import normalize
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

# ====================
data_name_list = ['europe', 'usa', 'film', 'actor', 'dblp', 'amazon', 'blogcatalog', 'ppi']
num_graphs = len(data_name_list)

# ====================
motif_stat = [] # List of graph-level emb (i.e., avg emb)
for t in range(num_graphs):
    # ====================
    data_name = data_name_list[t]
    # ==========
    pkl_file = open('emb/motif_null_stat_%s.pickle' % (data_name), 'rb')
    stat = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    #d = stat.shape[0]
    #for j in range(d):
    #    if stat[j] == 0.0:
    #        stat[j] = 1.0
    motif_stat.append(stat.reshape((1, -1)))
    print(stat)
    print(stat.shape)

# ====================
motif_stat = np.concatenate(motif_stat, axis=0)
# ==========
#motif_stat = normalize(motif_stat, 'l2', axis=0)
#motif_stat = normalize(motif_stat, 'l2')
# ==========
_, feat_dim = motif_stat.shape
for j in range(feat_dim):
    crt_mean = np.mean(motif_stat[:, j])
    crt_std = np.std(motif_stat[:, j])
    if crt_std > 0:
        motif_stat[:, j] = (motif_stat[:, j] - crt_mean) / crt_std
#motif_stat = normalize(motif_stat, 'l2')
# ==========
cor_mat = np.corrcoef(motif_stat)
print(cor_mat)

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

plt.title('SSP', fontsize=16, weight='bold', pad=15, loc='left')

# ====================
# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# ====================
plt.tight_layout()
plt.show()
plt.close()
