# Pre-compute PI-HIST (A) embedding for graph superfamily identification (GSI) on real graphs

import torch
import torch.nn.functional as F

import argparse
import gc
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
    parser.add_argument('--d', type=int, default=64) # 64
    parser.add_argument('--L', type=int, default=7)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--norm', type=str, default='no') # no, l2, z
    parser.add_argument('--act', type=str, default='no') # no, tanh, sig, relu, exp
    parser.add_argument('--phi_flag', type=bool, default=True)
    args = parser.parse_args()

    # ====================
    emb_dim = args.d
    RW_len = args.L
    eps = args.eps
    norm = args.norm
    act = args.act
    phi_flag = args.phi_flag

    data_name_list = ['europe', 'usa', 'film', 'actor', 'dblp', 'amazon', 'blogcatalog', 'ppi']
    num_RWs_list = [50000, 50000, 50000, 50000, 20000, 20000, 50000, 50000]
    num_graphs = len(data_name_list)

    # ====================
    num_nodes_list = []
    num_edges_list = []
    for data_name in data_name_list:
        # ====================
        pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
        edges = pickle.load(pkl_file)
        pkl_file.close()
        # ==========
        num_edges = len(edges)
        num_nodes = np.max(edges) + 1
        # ==========
        num_nodes_list.append(num_nodes)
        num_edges_list.append(num_edges)

    # ====================
    for t in range(num_graphs):
        # ====================
        data_name = data_name_list[t]
        num_RWs = num_RWs_list[t]
        num_nodes = num_nodes_list[t]

        # ====================
        pkl_file = open('AW_stat_GSIR/AW_hier_bth_%s_L=%d_n=%d.pickle'
                        % (data_name, RW_len, num_RWs), 'rb')
        bth_idxs = pickle.load(pkl_file)
        pkl_file.close()
        # ==========
        pkl_file = open('AW_stat_GSIR/AW_hier_src_%s_L=%d_n=%d.pickle'
                        % (data_name, RW_len, num_RWs), 'rb')
        src_idxs = pickle.load(pkl_file)
        pkl_file.close()
        # ==========
        pkl_file = open('AW_stat_GSIR/AW_hier_dst_%s_L=%d_n=%d.pickle'
                        % (data_name, RW_len, num_RWs), 'rb')
        dst_idxs = pickle.load(pkl_file)
        pkl_file.close()
        # ==========
        pkl_file = open('AW_stat_GSIR/AW_hier_vals_%s_L=%d_n=%d.pickle'
                        % (data_name, RW_len, num_RWs), 'rb')
        vals = pickle.load(pkl_file)
        pkl_file.close()

        # ====================
        hier_adj_list = []
        for r in range(RW_len):
            idxs_tnr = torch.LongTensor([bth_idxs[r],
                                         src_idxs[r],
                                         dst_idxs[r]])
            vals_tnr = torch.FloatTensor(vals[r])
            hier_adj = torch.sparse_coo_tensor(idxs_tnr,
                                               vals_tnr,
                                               size=[num_nodes, RW_len+1, RW_len+1]).to(device)
            hier_adj_list.append(hier_adj)
        del bth_idxs, src_idxs, dst_idxs, vals

        # ====================
        init_ide_emb = get_rand_proj_mat(RW_len+1, emb_dim, rand_seed=rand_seed_gbl)
        ide_emb_tnr = torch.FloatTensor(init_ide_emb).to(device)
        del init_ide_emb

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

        # ====================
        if torch.cuda.is_available():
            ide_emb = ide_emb_tnr.cpu().data.numpy()
        else:
            ide_emb = ide_emb_tnr.data.numpy()
        del ide_emb_tnr, hier_adj_list
        torch.cuda.empty_cache()
        gc.collect()

        # ====================
        # Save derived embedding
        pkl_file = open('GSIR_res/PI-HIST(A)_%s_L=%d_eps=%.1f_act=%s_norm=%s.pickle'
                        % (data_name, RW_len, eps, act, norm), 'wb')
        pickle.dump(ide_emb, pkl_file)
        pkl_file.close()
