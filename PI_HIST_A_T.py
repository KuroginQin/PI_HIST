# Efficiency evaluation for the training-free FFP of PI-HIST (A)

import torch
import torch.nn.functional as F

import gc
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

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str) # europe, usa, actor, film, ppi, blogcatalog, dblp, amazon
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--L', type=int, default=7)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--n', type=int, default=50000)
    parser.add_argument('--norm', type=str, default='no') # no, l2, z
    parser.add_argument('--act', type=str, default='no') # no, tanh, sig, relu, exp
    parser.add_argument('--nr', type=int, default=10) # Number of independent runs
    parser.add_argument('--phi_flag', type=bool, default=True)
    args = parser.parse_args()

    # ====================
    data_name = args.data_name
    emb_dim = args.d
    RW_len = args.L
    eps = args.eps
    num_RWs = args.n
    norm = args.norm
    act = args.act
    phi_flag = args.phi_flag
    if data_name == 'dblp': phi_flag = False
    num_runs = args.nr
    warmups = 5

    # ====================
    # Load graph topology
    pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
    edges = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    num_edges = len(edges)
    num_nodes = np.max(np.max(edges)) + 1
    print('#NODES %d #EDGES %d' % (num_nodes, num_edges))

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
    init_ide_emb = get_rand_proj_mat(RW_len+1, emb_dim, rand_seed=rand_seed_gbl)

    # ====================
    prop_time_list = []
    for iter in range(warmups+num_runs):
        # ====================
        if iter == warmups:
            print()

        # ====================
        torch.cuda.empty_cache()
        gc.collect()
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
        ide_emb_tnr = torch.FloatTensor(init_ide_emb).to(device)
        # ==========
        torch.cuda.synchronize()

        # ====================
        # Identity Embedding Derivation
        time_s = timeit.default_timer()
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
        time_e = timeit.default_timer()
        prop_time = time_e - time_s

        # ====================
        time_s = timeit.default_timer()
        if norm == 'l2':
            ide_emb_tnr = F.normalize(ide_emb_tnr, dim=1, p=2)
        elif norm == 'z':
            z_mean = torch.mean(ide_emb_tnr, dim=0).reshape((1, -1))
            z_std = torch.std(ide_emb_tnr, dim=0).reshape((1, -1))
            ide_emb_tnr = (ide_emb_tnr - z_mean) / z_std
        time_e = timeit.default_timer()
        prop_time += time_e - time_s
        prop_time_list.append(prop_time)
        print('PROP TIME %.4f' % (prop_time))
        del ide_emb_tnr

    prop_time_mean = np.mean(prop_time_list[-num_runs:])
    prop_time_std = np.std(prop_time_list[-num_runs:])
    print('PI-HIST (A) PROP TIME %.4f~(%.4f)' % (prop_time_mean, prop_time_std))
