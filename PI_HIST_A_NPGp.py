# Parameter analysis & ablation study of PI-HIST (A) on datasets w/ position ground-truth
# Conductance of node identity clustering

from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F

import argparse
import random
import pickle
import timeit
from utils import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

rand_seed_gbl = 0
setup_seed(rand_seed_gbl)

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_cond_mtc(edges, clus_res, num_clus):
    '''
    Function to get conductance metric w.r.t. a clustering result
    :param edges: edge list (undirected & 0-base node indices)
    :param clus_res: clustering result
    :param num_clus: number of clusters
    :return:
    '''
    # ====================
    cuts = [0.0 for _ in range(num_clus)]
    vols = [0.0 for _ in range(num_clus)]
    # ==========
    for (src, dst) in edges:
        # ==========
        src_lbl = clus_res[src]
        dst_lbl = clus_res[dst]
        # ==========
        vols[src_lbl] += 1.0
        vols[dst_lbl] += 1.0
        # ==========
        if src_lbl != dst_lbl:
            cuts[src_lbl] += 1.0
            cuts[dst_lbl] += 1.0
    # ==========
    cond = 0.0
    for c in range(num_clus):
        if vols[c] == 0:
            cond += 1.0
        else:
            cond += cuts[c] / vols[c]
    cond /= num_clus

    return cond

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str) # ppi
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--n', type=int, default=50000)  # 50000
    parser.add_argument('--norm', type=str, default='no') # no, l2, z
    parser.add_argument('--act', type=str, default='no') # no, tanh, sig, relu, exp
    parser.add_argument('--phi_flag', type=bool, default=True)
    args = parser.parse_args()

    # ====================
    data_name = args.data_name
    emb_dim = args.d
    num_RWs = args.n
    norm = args.norm
    act = args.act
    phi_flag = args.phi_flag

    RW_len_list = [5, 6, 7, 8, 9] # 5 choices
    eps_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 10 choices
    num_clus = 10

    # ====================
    # Load graph topology
    pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
    edges = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    # Load top-K similarity graph w.r.t. high-order deg feat
    pkl_file = open('data/%s_deg_sim_.pickle' % (data_name), 'rb')
    deg_sim_sp = pickle.load(pkl_file)
    pkl_file.close()
    # ====================
    # Load gnd (for node classifiction)
    pkl_file = open('data/%s_gnd.pickle' % (data_name), 'rb')
    gnd_sp = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    node_set = set()
    clas_set = set()
    for (node_idx, clas_idx) in gnd_sp:
        if node_idx not in node_set:
            node_set.add(node_idx)
        if clas_idx not in clas_set:
            clas_set.add(clas_idx)
    num_edges = len(edges)
    num_nodes = np.max(np.max(edges)) + 1
    num_clas = max(clas_set) + 1
    num_clus = min(num_clus, num_clas)
    # ==========
    gnd = np.zeros((num_nodes, num_clas))
    for (node_idx, clus_idx) in gnd_sp:
        gnd[node_idx, clus_idx] = 1.0
    print('#NODES %d #EDGES %d #CLUS %d' % (num_nodes, num_edges, num_clus))

    # ====================
    cond_res = []  # List of (mean) conductance
    for RW_len in RW_len_list:
        for eps in eps_list:
            # ====================
            pkl_file = open('AW_hier/AW_hier_bth_%s_L=%d_n=%d.pickle'
                            % (data_name, RW_len, num_RWs), 'rb')
            adj_bth_idxs = pickle.load(pkl_file)
            pkl_file.close()
            # ==========
            pkl_file = open('AW_hier/AW_hier_src_%s_L=%d_n=%d.pickle'
                            % (data_name, RW_len, num_RWs), 'rb')
            adj_src_idxs = pickle.load(pkl_file)
            pkl_file.close()
            # ==========
            pkl_file = open('AW_hier/AW_hier_dst_%s_L=%d_n=%d.pickle'
                            % (data_name, RW_len, num_RWs), 'rb')
            adj_dst_idxs = pickle.load(pkl_file)
            pkl_file.close()
            # ==========
            pkl_file = open('AW_hier/AW_hier_vals_%s_L=%d_n=%d.pickle'
                            % (data_name, RW_len, num_RWs), 'rb')
            adj_vals = pickle.load(pkl_file)
            pkl_file.close()

            # ====================
            hier_adj_list = []
            for r in range(RW_len):
                idxs_tnr = torch.LongTensor([adj_bth_idxs[r],
                                             adj_src_idxs[r],
                                             adj_dst_idxs[r]])
                vals_tnr = torch.FloatTensor(adj_vals[r])
                hier_adj = torch.sparse_coo_tensor(idxs_tnr,
                                                   vals_tnr,
                                                   size=[num_nodes, RW_len+1, RW_len+1]).to(device)
                hier_adj_list.append(hier_adj)

            # ====================
            init_ide_emb = get_rand_proj_mat(RW_len+1, emb_dim, rand_seed=rand_seed_gbl)
            ide_emb_tnr = torch.FloatTensor(init_ide_emb).to(device)

            # ====================
            # Identity Embedding Derivation
            for r in range(RW_len-1, -1, -1):
                # ==========
                hier_adj_d = hier_adj_list[r].to_dense()
                ide_emb_tnr = eps*ide_emb_tnr + (1-eps)*torch.matmul(hier_adj_d, ide_emb_tnr)
                del hier_adj_d
                # ==========
                if phi_flag:
                    for k in range(r+1):
                        for j in range(emb_dim):
                            z_mean = torch.mean(ide_emb_tnr[:, k, j])
                            z_std = torch.std(ide_emb_tnr[:, k, j])
                            if z_std > 0:
                                ide_emb_tnr[:, k, j] = (ide_emb_tnr[:, k, j] - z_mean) / z_std
            ide_emb_tnr = ide_emb_tnr[:, 0, :]
            if act == 'tanh':
                ide_emb_tnr = torch.tanh(ide_emb_tnr)
            elif act == 'sig':
                ide_emb_tnr = torch.sigmoid(ide_emb_tnr)
            elif act == 'relu':
                ide_emb_tnr = torch.relu(ide_emb_tnr)
            elif act == 'exp':
                ide_emb_tnr = torch.exp(ide_emb_tnr)

            # ====================
            #ide_emb_tnr = torch.nan_to_num(ide_emb_tnr)
            if norm == 'l2':
                ide_emb_tnr = F.normalize(ide_emb_tnr, dim=1, p=2)
            elif norm == 'z':
                z_mean = torch.mean(ide_emb_tnr, dim=0).reshape((1, -1))
                z_std = torch.std(ide_emb_tnr, dim=0).reshape((1, -1))
                ide_emb_tnr = (ide_emb_tnr - z_mean) / z_std
                ide_emb_tnr = torch.nan_to_num(ide_emb_tnr)
            # ==========
            if torch.cuda.is_available():
                ide_emb = ide_emb_tnr.cpu().data.numpy()
            else:
                ide_emb = ide_emb_tnr.data.numpy()

            # ====================
            print('PI-HIST(A) d=%d L=%d EPS=%f; act=%s norm=%s' % (emb_dim, RW_len, eps, act, norm))
            cond_list = []
            for rand_seed in range(10):
                kmeans = KMeans(n_clusters=num_clus, random_state=rand_seed).fit(ide_emb.astype(np.float64))
                clus_res = kmeans.labels_
                cond = get_cond_mtc(deg_sim_sp, clus_res, num_clus)
                cond_list.append(cond)
                # print('CLUS SEED=%d COND %.2f' % (rand_seed, cond*100))
            cond_mean = np.mean(cond_list)
            cond_std = np.std(cond_list)
            print('CLUS K=%d COND %.2f~(%.2f)' % (num_clus, cond_mean*100, cond_std*100))
            print()
            f_output = open('res/PI-HIST(A)_NPGp_%s_d=%d_act=%s_norm=%s.txt' % (data_name, emb_dim, act, norm), 'a+')
            f_output.write('PI-HIST(A) d=%d L=%d EPS=%f\n' % (emb_dim, RW_len, eps))
            f_output.write('CLUS K=%d COND %.2f~(%.2f)\n\n' % (num_clus, cond_mean*100, cond_std*100))
            f_output.close()
            # ==========
            cond_res.append(cond_mean)
    # ====================
    # Arrange results in matrix forms
    num_RW_len_sets = len(RW_len_list)
    num_eps_sets = len(eps_list)
    # ==========
    cond_res = np.array(cond_res)
    cond_res = np.reshape(cond_res, (num_RW_len_sets, num_eps_sets))
    # ====================
    #pkl_file = open('res/PI-HIST(A)_NPGp_cond_%s_d=%d_act=%s_norm=%s.pickle'
    #                % (data_name, emb_dim, act, norm), 'wb')
    #pickle.dump(cond_res, pkl_file)
    #pkl_file.close()

