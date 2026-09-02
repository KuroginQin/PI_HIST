# Pre-compute PI-HIST (R) embedding for graph superfamily identification (GSI) on real graphs

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

def get_GNN_sup_sp(edges, degs):
    # ====================
    src_idxs = []
    dst_idxs = []
    vals = []
    # ==========
    for (src, dst) in edges:
        # ==========
        src_idxs.append(src)
        dst_idxs.append(dst)
        vals.append(1.0 / (degs[src]))
        # ==========
        src_idxs.append(dst)
        dst_idxs.append(src)
        vals.append(1.0 / (degs[dst]))

    return src_idxs, dst_idxs, vals

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
    num_graphs = len(data_name_list) # Number of real graphs

    # =====================
    # Load data & pre-compute GNN support
    sups = []
    for data_name in data_name_list:
        # ====================
        pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
        edges = pickle.load(pkl_file)
        pkl_file.close()
        # ==========
        num_edges = len(edges)
        num_nodes = np.max(edges) + 1
        # ==========
        degs = [0.0 for _ in range(num_nodes)]
        for (src, dst) in edges:
            degs[src] += 1.0
            degs[dst] += 1.0
        # ==========
        sup_src_idxs, sup_dst_idxs, sup_vals = get_GNN_sup_sp(edges, degs)
        idxs_tnr = torch.LongTensor([sup_src_idxs, sup_dst_idxs])
        vals_tnr = torch.FloatTensor(sup_vals)
        sup_tnr = torch.sparse_coo_tensor(idxs_tnr, vals_tnr, size=[num_nodes, num_nodes])
        # ==========
        sups.append(sup_tnr)

    # ====================
    for t in range(num_graphs):
        # ====================
        sup = sups[t]
        data_name = data_name_list[t]

        # ====================
        num_nodes, _ = sup.shape
        sup_tnr = sup.to(device)
        init_pos_emb = get_rand_proj_mat(num_nodes, emb_dim, rand_seed=rand_seed_gbl)
        pos_emb_tnr = torch.FloatTensor(init_pos_emb).to(device)

        # ====================
        for _ in range(RW_len):
            pos_emb_tnr = eps*pos_emb_tnr + (1-eps)*torch.spmm(sup_tnr, pos_emb_tnr)
            # ==========
            if phi_flag:
                z_mean = torch.mean(pos_emb_tnr, dim=0).reshape((1, -1))
                z_std = torch.std(pos_emb_tnr, dim=0).reshape((1, -1))
                pos_emb_tnr = (pos_emb_tnr - z_mean) / z_std
        # ==========
        if act == 'tanh':
            pos_emb_tnr = torch.tanh(pos_emb_tnr)
        elif act == 'sig':
            pos_emb_tnr = torch.sigmoid(pos_emb_tnr)
        elif act == 'relu':
            pos_emb_tnr = torch.relu(pos_emb_tnr)
        elif act == 'exp':
            pos_emb_tnr = torch.exp(pos_emb_tnr)

        # ====================
        if norm == 'l2':
            pos_emb_tnr = F.normalize(pos_emb_tnr, dim=1, p=2)
        elif norm == 'z':
            z_mean = torch.mean(pos_emb_tnr, dim=0).reshape((1, -1))
            z_std = torch.std(pos_emb_tnr, dim=0).reshape((1, -1))
            pos_emb_tnr = (pos_emb_tnr - z_mean) / z_std

        # ====================
        if torch.cuda.is_available():
            pos_emb = pos_emb_tnr.cpu().data.numpy()
        else:
            pos_emb = pos_emb_tnr.data.numpy()
        del pos_emb_tnr, sup_tnr
        torch.cuda.empty_cache()
        gc.collect()

        # ====================
        # Save derived embedding
        pkl_file = open('emb/PI-HIST(R)_%s_L=%d_eps=%.1f_act=%s_norm=%s.pickle'
                        % (data_name, RW_len, eps, act, norm), 'wb')
        pickle.dump(pos_emb, pkl_file)
        pkl_file.close()
