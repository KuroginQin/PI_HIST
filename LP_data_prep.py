# Data split of all the real graph datasets for link prediction

import pickle
import random
import numpy as np
import networkx as nx
import timeit

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

rand_seed_gbl = 0
setup_seed(rand_seed_gbl)

# ====================
data_name = 'europe' # europe, usa, ppi, actor, blogcatalog, film, dblp, amazon
tst_ratio = 0.2
num_runs = int(1.0/tst_ratio) # Number of independent runs

# ====================
# Load graph topology
pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
edges = pickle.load(pkl_file)
pkl_file.close()
num_nodes = np.max(np.max(edges)) + 1
num_edges = len(edges)
print('#NODES %d #EDGES %d' % (num_nodes, num_edges))
# ==========
# Extract preserved edges vis DFS
# To ensure the masked topo has only 1 connected component
G = nx.Graph(edges)
psv_edges_list = set(nx.dfs_edges(G))
psv_edges_set = set()
for (src, dst) in psv_edges_list:
    if src > dst:
        tmp = src
        src = dst
        dst = tmp
    psv_edges_set.add((src, dst))
# ==========
acc_adj_list = [set() for _ in range(num_nodes)]
for (src, dst) in edges:
    acc_adj_list[src].add(dst)
    acc_adj_list[dst].add(src)

# ====================
# Randomly shuffle edge list
edges_shuf = []
for (src, dst) in edges:
    if (src, dst) not in psv_edges_set:
        edges_shuf.append((src, dst))
random.shuffle(edges_shuf)
num_ttl_tst_edges = len(edges_shuf)
num_tst_edges = int(num_ttl_tst_edges*tst_ratio) # Number of test edges in each independent run

# =====================
# Sample negative pairs
time_s = timeit.default_timer()
neg_gbl = []
for _ in range(num_ttl_tst_edges):
    u = random.randint(0, num_nodes-1)
    while True:
        v = random.randint(0, num_nodes-1)
        if u == v: continue
        elif u > v:
            tmp = u
            u = v
            v = tmp
        if v not in acc_adj_list[u]:
            acc_adj_list[u].add(v)
            acc_adj_list[v].add(u)
            neg_gbl.append((u, v))
            break
time_e = timeit.default_timer()
neg_smp_time = time_e - time_s
print('NEG SMP TIME', neg_smp_time)

# ====================
# Precompute test edges, negative pairs, & masked topo w.r.t. each independent run
trn_edges_list = []
trn_neg_list = []
tst_edges_list = []
tst_neg_list = []
for run_idx in range(num_runs):
    # ====================
    # Edges to be removed (i.e., test set) & negative pairs for evaluation
    if run_idx < num_runs-1:
        tst_edges = edges_shuf[run_idx*num_tst_edges: (run_idx+1)*num_tst_edges]
        tst_neg = neg_gbl[run_idx*num_tst_edges: (run_idx+1)*num_tst_edges]
    else:
        tst_edges = edges_shuf[run_idx*num_tst_edges:]
        tst_neg = neg_gbl[run_idx*num_tst_edges:]
    tst_edges_set = set(tst_edges)
    # ==========
    # Edges to be preserved (i.e., training set)
    trn_edges = []
    for (src, dst) in edges:
        if (src, dst) not in tst_edges_set:
            trn_edges.append((src, dst))

    # ====================
    num_trn = len(trn_edges)
    pos_set = set()
    for (src, dst) in trn_edges:
        if (src, dst) not in pos_set:
            pos_set.add((src, dst))
    trn_neg = []
    for _ in range(num_trn):
        u = random.randint(0, num_nodes - 1)
        while True:
            v = random.randint(0, num_nodes - 1)
            if u == v:
                continue
            elif u > v:
                tmp = u
                u = v
                v = tmp
            if (u, v) not in pos_set:
                trn_neg_list.append((u, v))
                break
    # ====================
    trn_edges_list.append(trn_edges)
    trn_neg_list.append(trn_neg)
    tst_edges_list.append(tst_edges)
    tst_neg_list.append(tst_neg)

# ====================
pkl_file = open('data_LP/%s_trn_edges_list.pickle' % (data_name), 'wb')
pickle.dump(trn_edges_list, pkl_file)
pkl_file.close()
# ==========
pkl_file = open('data_LP/%s_trn_neg_list.pickle' % (data_name), 'wb')
pickle.dump(trn_neg_list, pkl_file)
pkl_file.close()
# ==========
pkl_file = open('data_LP/%s_tst_edges_list.pickle' % (data_name), 'wb')
pickle.dump(tst_edges_list, pkl_file)
pkl_file.close()
# ==========
pkl_file = open('data_LP/%s_tst_neg_list.pickle' % (data_name), 'wb')
pickle.dump(tst_neg_list, pkl_file)
pkl_file.close()
