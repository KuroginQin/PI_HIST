# Visualization for Bag of degree (BoD)'s result of graph superfamily identification (GSI) on real graphs

import numpy as np
import networkx as nx
from collections import Counter
from typing import List

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

def get_deg_cnt(graph_list: List[nx.Graph]) -> np.ndarray:
    # ====================
    cnt_list = []
    for graph in graph_list:
        degs = []
        for node in graph.nodes:
            deg = graph.degree[node]
            degs.append(deg)
            #print('node', node, 'deg', deg)

        cnt = Counter()
        for d in degs:
            cnt[d] += 1
        cnt_list.append(cnt)

    # ====================
    # Merge stat to a feature matrix
    keys = sorted(set().union(*(feat.keys() for feat in cnt_list)))
    feat_mat = np.zeros((len(graph_list), len(keys)))
    # ==========
    for i, feat in enumerate(cnt_list):
        for j, key in enumerate(keys):
            feat_mat[i, j] = feat.get(key, 0)

    return feat_mat


# ====================
data_name_list = ['europe', 'usa', 'film', 'actor', 'dblp', 'amazon', 'blogcatalog', 'ppi']
num_graphs = len(data_name_list)

# =====================
graphs = []
for data_name in data_name_list:
    # ====================
    pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
    edges = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    G = nx.Graph()
    G.add_edges_from(edges)
    graphs.append(G)

# ====================
time_s = timeit.default_timer()
deg_cnt = get_deg_cnt(graphs)
time_e = timeit.default_timer()
feat_time = time_e - time_s
print('FEAT TIME %f' % (feat_time))

cor_mat = np.corrcoef(deg_cnt)
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

plt.title('BoD', fontsize=16, weight='bold', pad=15, loc='left')

# ====================
# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# ====================
plt.tight_layout()
plt.show()
plt.close()
