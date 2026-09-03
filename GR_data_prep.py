# Data split of all the real graph datasets for graph recontruction

import numpy as np
import random
import timeit
import math
import pickle

# ====================
data_name = 'europe' # europe, usa, ppi
smp_ratio = 1e-1
# ==========
#data_name = 'actor' # actor, blocatalog
#smp_ratio = 1e-2
# ==========
#data_name = 'film' # film
#smp_ratio = 1e-3
# ==========
#data_name = 'dblp' # dblp, amazon
#smp_ratio = 1e-5

seed_list = [i for i in range(10)]

# ====================
pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
edges = pickle.load(pkl_file)
pkl_file.close()
# ==========
num_nodes = np.max(edges) + 1
num_edges = len(edges)
print('#NODES %d #EDGES %d' % (num_nodes, num_edges))

num_pairs = num_nodes/2*num_nodes - num_nodes/2
num_smp = num_pairs*smp_ratio
print('#PAIRS %d #SMP %d' % (num_pairs, num_smp))

# ====================
edge_set = set()
for (src, dst) in edges:
    # ==========
    if src == dst:
        continue
    if src >  dst:
        tmp = src
        src = dst
        dst = tmp
    # ==========
    if (src, dst) not in edge_set:
        edge_set.add((src, dst))

# ====================
run_idx = 0
num_runs = len(seed_list)
pos_pairs_list = []
neg_pairs_list = []
for rand_seed in seed_list:
    # ====================
    print('%d / %d' % (run_idx+1, num_runs))
    time_s = timeit.default_timer()
    np.random.seed(rand_seed)
    # ==========
    #gbl_idxs = np.random.choice(int(num_pairs), size=int(num_smp), replace=False)
    # ==========
    selected_idxs = set()
    while len(selected_idxs) < num_smp:
        idx = random.randint(0, num_pairs - 1)
        if idx not in selected_idxs:
            selected_idxs.add(idx)
    gbl_idxs = np.array(list(selected_idxs))
    # ==========
    time_e = timeit.default_timer()
    rand_smp_time = time_e - time_s
    print('RAND SMP TIME %f' % (rand_smp_time))
    # ==========
    time_s = timeit.default_timer()
    pairs = []
    for k in gbl_idxs:
        # ==========
        '''
        i = 0
        k_temp = k
        while k_temp >= num_nodes - i - 1:
            k_temp -= num_nodes - i - 1
            i += 1
        j = i + 1 + k_temp
        '''
        # ==========
        i = int(math.floor((2*num_nodes - 1 - math.sqrt((2*num_nodes - 1)**2 - 8*k)) / 2))
        start_index = i*(2*num_nodes - i - 1) // 2
        j = i + 1 + (k - start_index)
        # ===========
        pairs.append((int(i), int(j)))
    time_e = timeit.default_timer()
    pair_ext_time = time_e - time_s
    print('PAIR EXT TIME %f' % (pair_ext_time))

    # ====================
    time_s = timeit.default_timer()
    pos_pairs = []
    neg_pairs = []
    for (src, dst) in pairs:
        # ==========
        if src == dst: continue
        if src > dst:
            tmp = src
            src = dst
            dst = tmp
        # ==========
        if (src, dst) in edge_set:
            pos_pairs.append((src, dst))
        else:
            neg_pairs.append((src, dst))
    # ==========
    pos_pairs = sorted(pos_pairs)
    neg_pairs = sorted(neg_pairs)
    # ==========
    pos_pairs_list.append(pos_pairs)
    neg_pairs_list.append(neg_pairs)
    time_e = timeit.default_timer()
    split_time = time_e - time_s
    print('SPLIT TIME %f' % (split_time))
    # ==========
    run_idx += 1

# ====================
pkl_file = open('data_LP/GR_%s_pos.pickle' % (data_name), 'wb')
pickle.dump(pos_pairs_list, pkl_file)
pkl_file.close()
# ==========
pkl_file = open('data_LP/GR_%s_neg.pickle' % (data_name), 'wb')
pickle.dump(neg_pairs_list, pkl_file)
pkl_file.close()

