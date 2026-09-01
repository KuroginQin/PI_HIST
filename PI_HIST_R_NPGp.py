# Parameter analysis & ablation study of PI-HIST (R) on datasets w/ position ground-truth
# Macro-F1 & Micro-F1 of node position classification on validation set

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F

import argparse
import random
import pickle
import timeit
import scipy.sparse as sp

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

def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = y.sum(axis=1, dtype=np.int32)
    num_label = np.reshape(num_label, (-1, 1))
    #num_label = np.sum(y, axis=1, dtype=np.int)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    #y_pred = np.zeros_like(y_score, dtype=np.int32)
    row, col = [], []
    for i in range(y_score.shape[0]):
        row += [i]*num_label[i, 0]
        col += y_sort[i, :num_label[i, 0]].tolist()
        #for j in range(num_label[i, 0]):
        #    y_pred[i, y_sort[i, j]] = 1
    y_pred = sp.csr_matrix(
            ([1]*len(row), (row, col)),
            shape=y.shape, dtype=np.bool_)

    return y_pred

def NPC_eva(emb, gnd, trn_ratio=0.2, val_ratio=0.1, n_splits=10, random_state=0, C=1.):
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
        gnd_trn, gnd_val, gnd_tst = gnd[trn_idx], gnd[val_idx], gnd[tst_idx]
        # ====================
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    #multi_class="ovr",
                    max_iter=5000),
                n_jobs=-1)
        clf.fit(emb_trn, gnd_trn)
        # ==========
        score = clf.predict_proba(emb_val)
        pred_val = construct_indicator(score, gnd_val)
        mi = f1_score(gnd_val, pred_val, average="micro")
        ma = f1_score(gnd_val, pred_val, average="macro")
        micro_val.append(mi)
        macro_val.append(ma)
        # ==========
        score = clf.predict_proba(emb_tst)
        pred_tst = construct_indicator(score, gnd_tst)
        mi = f1_score(gnd_tst, pred_tst, average="micro")
        ma = f1_score(gnd_tst, pred_tst, average="macro")
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
    parser.add_argument('--data_name', type=str) # ppi
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--norm', type=str, default='no') # no, l2, z
    parser.add_argument('--act', type=str, default='no') # no, tanh, sig, relu, exp
    parser.add_argument('--phi_flag', type=bool, default=True)
    args = parser.parse_args()

    # ====================
    data_name = args.data_name
    emb_dim = args.d
    norm = args.norm
    act = args.act
    phi_flag = args.phi_flag

    RW_len_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] # 16 choices
    eps_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 10 choices
    tr_ratio = 0.2

    # ====================
    # Load graph topology
    pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
    edges = pickle.load(pkl_file)
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
    # ==========
    gnd = np.zeros((num_nodes, num_clas))
    for (node_idx, clus_idx) in gnd_sp:
        gnd[node_idx, clus_idx] = 1.0
    print('#NODES %d #EDGES %d #CLAS %d' % (num_nodes, num_edges, num_clas))

    # ====================
    degs = [0.0 for _ in range(num_nodes)]
    for (src, dst) in edges:
        degs[src] += 1.0
        degs[dst] += 1.0
    # ====================
    # Position Embedding Derivation
    sup_src_idxs, sup_dst_idxs, sup_vals = get_GNN_sup_sp(edges, degs)
    idxs_tnr = torch.LongTensor([sup_src_idxs, sup_dst_idxs])
    vals_tnr = torch.FloatTensor(sup_vals)
    sup_tnr = torch.sparse_coo_tensor(idxs_tnr, vals_tnr, size=[num_nodes, num_nodes]).to(device)

    # ====================
    mi_val_res = [] # List of (mean) Micro-F1 on validation set
    ma_val_res = [] # List of (mean) Macro-F1 on validation set
    for RW_len in RW_len_list:
        for eps in eps_list:
            # ====================
            init_pos_emb = get_rand_proj_mat(num_nodes, emb_dim, rand_seed=rand_seed_gbl)
            pos_emb_tnr = torch.FloatTensor(init_pos_emb).to(device)

            # ====================
            for _ in range(RW_len):
                pos_emb_tnr = eps*pos_emb_tnr + (1-eps) * torch.spmm(sup_tnr, pos_emb_tnr)
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
            # ==========
            if torch.cuda.is_available():
                pos_emb = pos_emb_tnr.cpu().data.numpy()
            else:
                pos_emb = pos_emb_tnr.data.numpy()

            # ====================
            print('PI-HIST(R) d=%d L=%d EPS=%f' % (emb_dim, RW_len, eps))
            eva_res = NPC_eva(pos_emb, gnd, trn_ratio=tr_ratio, n_splits=10, random_state=rand_seed_gbl)
            mi_mean_val = eva_res[0]
            mi_std_val = eva_res[1]
            ma_mean_val = eva_res[2]
            ma_std_val = eva_res[3]
            val_ratio = 0.1
            print('NPC VAL-%.1f MICRO-F1 %.2f~(%.2f) MACRO-F1 %.2f~(%.2f)'
                  % (val_ratio, mi_mean_val*100, mi_std_val*100, ma_mean_val*100, ma_std_val*100))
            print()
            f_output = open('res/PI-HIST(R)_NPGp_%s_d=%d_act=%s_norm=%s.txt' % (data_name, emb_dim, act, norm), 'a+')
            f_output.write('PI-HIST(R) d=%d L=%d EPS=%f\n' % (emb_dim, RW_len, eps))
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
    #pkl_file = open('res/PI-HIST(R)_NPGp_mi_%s_d=%d_act=%s_norm=%s.pickle'
    #                % (data_name, emb_dim, act, norm), 'wb')
    #pickle.dump(mi_val_res, pkl_file)
    #pkl_file.close()
    # ==========
    #pkl_file = open('res/PI-HIST(R)_NPGp_ma_%s_d=%d_act=%s_norm=%s.pickle'
    #                % (data_name, emb_dim, act, norm), 'wb')
    #pickle.dump(ma_val_res, pkl_file)
    #pkl_file.close()
