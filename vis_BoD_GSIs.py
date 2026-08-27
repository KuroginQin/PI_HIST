# Visualization for Bag of degree (BoD)'s result of graph superfamily identification (GSI) on synthetic graphs

import numpy as np
import networkx as nx
from collections import Counter
from typing import List

import pickle
import timeit
import matplotlib.pyplot as plt
import seaborn as sns


def get_deg_cnt(graph_list: List[nx.Graph]) -> np.ndarray:
    # ====================
    # Get deg count stat for each node
    cnt_list = []
    for graph in graph_list:
        degs = []
        for node in graph.nodes:
            deg = graph.degree[node]
            degs.append(deg)

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

# =====================
# List of graphs for all runs
graphs_list = [[] for _ in range(num_runs)]
# =====================
for c_min_ in c_min_list:
    # ====================
    pkl_file = open('data/LFR_edges_list_n=%d_mu=%.1f_k=%d_maxk=%d_minc=%d_maxc=%d.pickle'
                    % (num_nodes, mu, k_avg, k_max, c_min_, c_max), 'rb')
    edges_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    for t in range(num_runs):
        # ==========
        # Graph of the (t+1)-th run
        edges = edges_list[t]
        G = nx.Graph()
        G.add_edges_from(edges)
        graphs_list[t].append(G)
# ====================
for k_avg_ in k_avg_list:
    if k_avg_ == k_avg_list[0]: continue # Skip the repeated graph
    # ====================
    pkl_file = open('data/LFR_edges_list_n=%d_mu=%.1f_k=%d_maxk=%d_minc=%d_maxc=%d.pickle'
                    % (num_nodes, mu, k_avg_, k_max, c_min, c_max), 'rb')
    edges_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    for t in range(num_runs):
        # ==========
        # Graph of the (t+1)-th run
        edges = edges_list[t]
        G = nx.Graph()
        G.add_edges_from(edges)
        graphs_list[t].append(G)

# =====================
# Average correlation matrix
avg_cor_mat = np.zeros((num_graphs, num_graphs))
for t in range(num_runs):
    # ====================
    graphs = graphs_list[t]
    # ==========
    # Get BoD features of current run
    time_s = timeit.default_timer()
    deg_cnt = get_deg_cnt(graphs)
    time_e = timeit.default_timer()
    feat_time = time_e - time_s
    # ==========
    print('RUN-%d FEAT TIME %f' % (t+1, feat_time))

    # ====================
    # Get correlation matrix of current run
    cor_mat = np.corrcoef(deg_cnt)
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
                 xticklabels=[r'$G_{%d}$' % (i) for i in range(1, num_graphs + 1)],
                 yticklabels=[r'$G_{%d}$' % (i) for i in range(1, num_graphs + 1)]
                 )
ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)

plt.title('BoD', fontsize=16, weight='bold', pad=15, loc='left')

# ====================
# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# ====================
plt.tight_layout()
plt.show()
# plt.savefig('vis/BoD_GSIs.svg',
#            format='svg',
#            bbox_inches='tight',
#            transparent=False)
plt.close()
