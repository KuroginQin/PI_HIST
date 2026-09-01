# Parameter analysis & ablation study of PI-HIST (A) on datasets w/ identity ground-truth
# Macro-F1 & Micro-F1 of node identity classification on validation set

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F

import argparse
import random
import pickle
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

def NIC_eva(emb, gnd, trn_ratio=0.2, val_ratio=0.1, n_splits=10, random_state=0):
    # ====================
    n, d = emb.shape
    num_val = int(val_ratio*n)
    micro_val, macro_val, micro_tst, macro_tst = [], [], [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-trn_ratio, random_state=random_state)
    # ====================
    for trn_idx, tst_idx in shuffle.split(emb):
        # ====================
        val_idx = tst_idx[:num_val]
        tst_idx = tst_idx[num_val:]
        emb_trn, emb_val, emb_tst = emb[trn_idx], emb[val_idx], emb[tst_idx]
        gnd_trn, gnd_val, gnd_tst = np.array(gnd)[trn_idx], np.array(gnd)[val_idx], np.array(gnd)[tst_idx]
        # ====================
        clf = LogisticRegression(solver='lbfgs', max_iter=5000, random_state=random_state)
        clf.fit(emb_trn, gnd_trn)
        # ==========
        clf_val_res = clf.predict(emb_val)
        mi = f1_score(gnd_val, clf_val_res, average="micro", zero_division=1)
        ma = f1_score(gnd_val, clf_val_res, average="macro", zero_division=1)
        micro_val.append(mi)
        macro_val.append(ma)
        # ==========
        clf_tst_res = clf.predict(emb_tst)
        mi = f1_score(gnd_tst, clf_tst_res, average="micro", zero_division=1)
        ma = f1_score(gnd_tst, clf_tst_res, average="macro", zero_division=1)
        micro_tst.append(mi)
        macro_tst.append(ma)

    # ====================
    mi_mean_val = np.mean(micro_val)
    mi_std_val = np.std(micro_val)
    ma_mean_val = np.mean(macro_val)
    ma_std_val = np.std(macro_val)
    # ====================
    mi_mean_tst = np.mean(micro_tst)
    mi_std_tst = np.std(micro_tst)
    ma_mean_tst = np.mean(macro_tst)
    ma_std_tst = np.std(macro_tst)

    return (mi_mean_val, mi_std_val, ma_mean_val, ma_std_val,
            mi_mean_tst, mi_std_tst, ma_mean_tst, ma_std_tst)

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str) # usa
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--n', type=int, default=50000) # 50000
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
    tr_ratio = 0.2

    # ====================
    pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
    edges = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    # Load gnd (for node classification)
    pkl_file = open('data/%s_gnd.pickle' % (data_name), 'rb')
    gnd = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    num_nodes = len(gnd)
    num_edges = len(edges)
    num_clas = np.max(gnd) + 1
    print('#NODES %d #EDGES %d #CLAS %d' % (num_nodes, num_edges, num_clas))

    # ====================
    mi_val_res = [] # List of (mean) Micro-F1 on validation set
    ma_val_res = [] # List of (mean) Macro-F1 on validation set
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
                ide_emb_tnr = eps*ide_emb_tnr + (1-eps) * torch.matmul(hier_adj_d, ide_emb_tnr)
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
            eva_res = NIC_eva(ide_emb, gnd, trn_ratio=tr_ratio, n_splits=10, random_state=rand_seed_gbl)
            mi_mean_val = eva_res[0]
            mi_std_val = eva_res[1]
            ma_mean_val = eva_res[2]
            ma_std_val = eva_res[3]
            val_ratio = 0.1
            print('NIC VAL-%.1f MICRO-F1 %.2f~(%.2f) MACRO-F1 %.2f~(%.2f)'
                  % (val_ratio, mi_mean_val*100, mi_std_val*100, ma_mean_val*100, ma_std_val*100))
            print()
            f_output = open('res/PI-HIST(A)_NIGp_%s_d=%d_act=%s_norm=%s.txt' % (data_name, emb_dim, act, norm), 'a+')
            f_output.write('PI-HIST(A) d=%d L=%d EPS=%f\n' % (emb_dim, RW_len, eps))
            f_output.write('NIC VAL-%.1f MICRO-F1 %.2f~(%.2f) MACRO-F1 %.2f~(%.2f)\n\n'
                           % (val_ratio, mi_mean_val*100, mi_std_val*100, ma_mean_val*100, ma_std_val*100))
            f_output.close()
            # ==========
            mi_val_res.append(mi_mean_val)
            ma_val_res.append(ma_mean_val)

    # ====================
    # Arrange results in matrix forms
    num_RW_len_sets = len(RW_len_list)
    num_eps_sets = len(eps_list)
    # ==========
    mi_val_res = np.array(mi_val_res)
    mi_val_res = np.reshape(mi_val_res, (num_RW_len_sets, num_eps_sets))
    # ==========
    ma_val_res = np.array(ma_val_res)
    ma_val_res = np.reshape(ma_val_res, (num_RW_len_sets, num_eps_sets))
    # ====================
    #pkl_file = open('res/PI-HIST(A)_NIGp_mi_%s_d=%d_act=%s_norm=%s.pickle'
    #                % (data_name, emb_dim, act, norm), 'wb')
    #pickle.dump(mi_val_res, pkl_file)
    #pkl_file.close()
    # ==========
    #pkl_file = open('res/PI-HIST(A)_NIGp_ma_%s_d=%d_act=%s_norm=%s.pickle'
    #                % (data_name, emb_dim, act, norm), 'wb')
    #pickle.dump(ma_val_res, pkl_file)
    #pkl_file.close()
