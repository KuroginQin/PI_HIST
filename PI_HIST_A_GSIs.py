# Pre-compute PI-HIST (A) embedding for graph superfamily identification (GSI) on synthetic graphs

import torch
import torch.nn.functional as F

import gc
import argparse
import random
import pickle
#import timeit
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

    # ====================
    num_runs = 10  # Number of independent runs
    num_nodes = 1000  # Number of nodes in each syn graphs
    # ==========
    c_min = 100
    c_min_list = [20, 40, 60, 80, 100]  # G_1 - G_5
    c_max = 500
    # ==========
    d_avg = 10
    d_avg_list = [9, 8, 7, 6]  # G_6 - G_9
    d_max = 50
    # ==========
    mu = 0.1

    # ====================
    # Load pre-computed AW-induced hier struc
    pkl_file = open('AW_hier/AW_hier_bth_LFR_L=%d.pickle' % (RW_len), 'rb')
    bth_idxs_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier/AW_hier_src_LFR_L=%d.pickle' % (RW_len), 'rb')
    src_idxs_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier/AW_hier_dst_LFR_L=%d.pickle' % (RW_len), 'rb')
    dst_idxs_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier/AW_hier_vals_LFR_L=%d.pickle' % (RW_len), 'rb')
    vals_list = pickle.load(pkl_file)
    pkl_file.close()

    # ====================
    embs_list = [] # List of emb for all graphs in all runs
    for t in range(num_runs):
        embs = [] # List of node-wise embeddings for G1-G9 in current run
        num_graphs = len(bth_idxs_list[t]) # Number of syn graphs in current run
        for s in range(num_graphs):
            # ====================
            bth_idxs = bth_idxs_list[t][s]
            src_idxs = src_idxs_list[t][s]
            dst_idxs = dst_idxs_list[t][s]
            vals = vals_list[t][s]
            # ==========
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
            init_ide_emb = get_rand_proj_mat(RW_len + 1, emb_dim, rand_seed=rand_seed_gbl)
            ide_emb_tnr = torch.FloatTensor(init_ide_emb).to(device)
            del init_ide_emb

            # ====================
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
            embs.append(ide_emb)
        # ====================
        embs_list.append(embs)

    # ====================
    # Save derived embedding
    pkl_file = open('emb/PI-HIST(A)_LFR_L=%d_eps=%.1f_act=%s_norm=%s.pickle'
                    % (RW_len, eps, act, norm), 'wb')
    pickle.dump(embs_list, pkl_file)
    pkl_file.close()
