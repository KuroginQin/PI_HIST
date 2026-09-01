# Generate synthetic graphs for graph superfamily identification using the LFR benchmark

import os
import timeit
import pickle as pkl

# ====================
num_runs = 10 # Number of independent runs
num_nodes = 1000 # Number of nodes in each syn graphs
# ==========
c_min = 100
c_min_list = [20, 40, 60, 80, 100] # G_1 - G_5
c_max = 500
# ==========
d_avg = 10
d_avg_list = [9, 8, 7, 6] # G_6 - G_9
d_max = 50
# ==========
mu = 0.1

# =====================
for c_min_ in c_min_list:
    # =====================
    edges_list = []
    for t in range(num_runs):
        # =====================
        print('minc %d maxc %d k %d maxk %d mu %.2f RUN-%d' % (c_min_, c_max, d_avg, d_max, mu, t))
        time_s = timeit.default_timer()
        cmd = os.popen('LFR_gen/benchmark -N %d -minc %d -maxc %d -k %d -maxk %d -mu %.2f'
                       % (num_nodes, c_min_, c_max, d_avg, d_max, mu)).readlines()
        time_e = timeit.default_timer()
        gen_time = time_e - time_s
        print('GEN TIME %f' % (gen_time))

        # ====================
        # Load the generated graph topology
        node_cnt = 0
        node_map = {}
        edge_set = set()
        f_input = open('network.dat', 'r')
        for line in f_input.readlines():
            # ==========
            rec = line.strip().split('	')
            src = int(rec[0])
            dst = int(rec[1])
            if src == dst: continue
            # ==========
            if src not in node_map:
                node_map[src] = node_cnt
                src_idx = node_cnt
                node_cnt += 1
            else:
                src_idx = node_map[src]
            # ==========
            if dst not in node_map:
                node_map[dst] = node_cnt
                dst_idx = node_cnt
                node_cnt += 1
            else:
                dst_idx = node_map[dst]
            # ==========
            if src_idx > dst_idx:
                tmp = src_idx
                src_idx = dst_idx
                dst_idx = tmp
            if (src_idx, dst_idx) not in edge_set:
                edge_set.add((src_idx, dst_idx))
        f_input.close()
        # ====================
        edges = sorted(list(edge_set))
        num_edges = len(edges)
        edges_list.append(edges)
        # ==========
        print('#NODES %d #EDGES %d' % (num_nodes, num_edges))
        print()

        # ====================
        #pkl_file = open('data/LFR_edges_list_n=%d_mu=%.1f_k=%d_maxk=%d_minc=%d_maxc=%d.pickle'
        #                % (num_nodes, mu, d_avg, d_max, c_min_, c_max), 'wb')
        #pkl.dump(edges_list, pkl_file)
        #pkl_file.close()

# =====================
for d_avg_ in d_avg_list:
    # ====================
    edges_list = []
    for t in range(num_runs):
        # =====================
        print('minc %d maxc %d k %d maxk %d mu %.2f RUN-%d' % (c_min, c_max, d_avg_, d_max, mu, t))
        time_s = timeit.default_timer()
        cmd = os.popen('LFR_gen/benchmark -N %d -minc %d -maxc %d -k %d -maxk %d -mu %.2f'
                       % (num_nodes, c_min, c_max, d_avg_, d_max, mu)).readlines()
        time_e = timeit.default_timer()
        gen_time = time_e - time_s
        print('GEN TIME %f' % (gen_time))

        # ====================
        # Load the generated graph topology
        node_cnt = 0
        node_map = {}
        edge_set = set()
        f_input = open('network.dat', 'r')
        for line in f_input.readlines():
            # ==========
            rec = line.strip().split('	')
            src = int(rec[0])
            dst = int(rec[1])
            if src == dst: continue
            # ==========
            if src not in node_map:
                node_map[src] = node_cnt
                src_idx = node_cnt
                node_cnt += 1
            else:
                src_idx = node_map[src]
            # ==========
            if dst not in node_map:
                node_map[dst] = node_cnt
                dst_idx = node_cnt
                node_cnt += 1
            else:
                dst_idx = node_map[dst]
            # ==========
            if src_idx > dst_idx:
                tmp = src_idx
                src_idx = dst_idx
                dst_idx = tmp
            if (src_idx, dst_idx) not in edge_set:
                edge_set.add((src_idx, dst_idx))
        f_input.close()
        # ====================
        edges = sorted(list(edge_set))
        num_edges = len(edges)
        edges_list.append(edges)
        # ==========
        print('#NODES %d #EDGES %d' % (num_nodes, num_edges))
        print()

        # ====================
        #pkl_file = open('data/LFR_edges_list_n=%d_mu=%.1f_k=%d_maxk=%d_minc=%d_maxc=%d.pickle'
        #                % (num_nodes, mu, d_avg_, d_max, c_min, c_max), 'wb')
        #pkl.dump(edges_list, pkl_file)
        #pkl_file.close()

