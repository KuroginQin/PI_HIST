# Link prediction for PI-HIST (A) on all real graphs

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

def LP_eva(trn_emb_src, trn_emb_dst, val_emb_src, val_emb_dst, tst_emb_src, tst_emb_dst, trn_gnd, val_gnd, tst_gnd):
    # ====================
    # Binary Classifier - Concatenate
    trn_emb = np.concatenate((trn_emb_src, trn_emb_dst), axis=1)
    clas = LogisticRegression(random_state=rand_seed_gbl, solver='lbfgs')
    clas.fit(trn_emb, trn_gnd)
    del trn_emb
    # ====================
    val_emb = np.concatenate((val_emb_src, val_emb_dst), axis=1)
    probs = clas.predict_proba(val_emb)
    probs = probs[:, 1]
    del val_emb
    # ==========
    auc_val = roc_auc_score(val_gnd, probs)
    ap_val = average_precision_score(val_gnd, probs) # Average Precision
    median = np.median(probs)
    preds_ = []
    for prob in probs:
        if prob > median:
            preds_.append(1)
        else:
            preds_.append(0)
    # ====================
    tst_emb = np.concatenate((tst_emb_src, tst_emb_dst), axis=1)
    probs = clas.predict_proba(tst_emb)
    probs = probs[:, 1]
    del tst_emb
    # ==========
    auc_tst = roc_auc_score(tst_gnd, probs)
    ap_tst = average_precision_score(tst_gnd, probs) # Average Precision
    median = np.median(probs)
    preds_ = []
    for prob in probs:
        if prob > median:
            preds_.append(1)
        else:
            preds_.append(0)

    return auc_val, ap_val, auc_tst, ap_tst

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--L', type=int, default=7)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--n', type=int, default=50000)
    parser.add_argument('--norm', type=str, default='no') # no, l2, z
    parser.add_argument('--act', type=str, default='no') # no, tanh, sig, relu, exp
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
    # ==========
    val_ratio = 0.1
    tst_ratio = 0.1

    # ====================
    pkl_file = open('data_LP/%s_trn_edges_list.pickle' % (data_name), 'rb')
    trn_edges_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('data_LP/%s_trn_neg_list.pickle' % (data_name), 'rb')
    trn_neg_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('data_LP/%s_tst_edges_list.pickle' % (data_name), 'rb')
    tst_edges_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('data_LP/%s_tst_neg_list.pickle' % (data_name), 'rb')
    tst_neg_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    num_runs = len(trn_edges_list) # Number of independent runs

    # ====================
    pkl_file = open('AW_hier_LP/AW_hier_bth_%s_L=%d_n=%d.pickle' % (data_name, RW_len, num_RWs), 'rb')
    bth_idxs_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier_LP/AW_hier_src_%s_L=%d_n=%d.pickle' % (data_name, RW_len, num_RWs), 'rb')
    src_idxs_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier_LP/AW_hier_dst_%s_L=%d_n=%d.pickle' % (data_name, RW_len, num_RWs), 'rb')
    dst_idxs_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier_LP/AW_hier_vals_%s_L=%d_n=%d.pickle' % (data_name, RW_len, num_RWs), 'rb')
    vals_list = pickle.load(pkl_file)
    pkl_file.close()

    # ====================
    cat_auc_val = []
    cat_ap_val = []
    # ==========
    cat_auc_tst = []
    cat_ap_tst = []
    for t in range(num_runs):
        # ====================
        trn_edges = trn_edges_list[t]
        trn_neg = trn_neg_list[t]
        tst_edges = tst_edges_list[t]
        tst_neg = tst_neg_list[t]
        # ==========
        num_edges = len(trn_edges)
        num_nodes = np.max(np.max(trn_edges)) + 1
        # ==========
        n = len(tst_edges)
        num_val = int(val_ratio / (val_ratio+tst_ratio)*n)
        val_edges = tst_edges[:num_val]
        tst_edges = tst_edges[num_val:]
        val_neg = tst_neg[:num_val]
        tst_neg = tst_neg[num_val:]

        # ====================
        trn_src_idxs = []
        trn_dst_idxs = []
        trn_gnd = []
        # ==========
        val_src_idxs = []
        val_dst_idxs = []
        val_gnd = []
        # ==========
        tst_src_idxs = []
        tst_dst_idxs = []
        tst_gnd = []
        # ==========
        for (src, dst) in trn_edges:
            trn_src_idxs.append(src)
            trn_dst_idxs.append(dst)
            trn_gnd.append(1.0)
        # ==========
        for (src, dst) in trn_neg:
            trn_src_idxs.append(src)
            trn_dst_idxs.append(dst)
            trn_gnd.append(0.0)
        # ==========
        for (src, dst) in val_edges:
            val_src_idxs.append(src)
            val_dst_idxs.append(dst)
            val_gnd.append(1.0)
        # ==========
        for (src, dst) in val_neg:
            val_src_idxs.append(src)
            val_dst_idxs.append(dst)
            val_gnd.append(0.0)
        # ==========
        for (src, dst) in tst_edges:
            tst_src_idxs.append(src)
            tst_dst_idxs.append(dst)
            tst_gnd.append(1.0)
        # ==========
        for (src, dst) in tst_neg:
            tst_src_idxs.append(src)
            tst_dst_idxs.append(dst)
            tst_gnd.append(0.0)

        # ====================
        # PI-HIST (A) emb derivation
        adj_bth_idxs = bth_idxs_list[t]
        adj_src_idxs = src_idxs_list[t]
        adj_dst_idxs = dst_idxs_list[t]
        adj_vals = vals_list[t]
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
        del adj_bth_idxs, adj_src_idxs, adj_dst_idxs, adj_vals

        # ====================
        init_ide_emb = get_rand_proj_mat(RW_len+1, emb_dim, rand_seed=rand_seed_gbl)
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
        # ==========
        if torch.cuda.is_available():
            ide_emb = ide_emb_tnr.cpu().data.numpy()
        else:
            ide_emb = ide_emb_tnr.data.numpy()
        del ide_emb_tnr, hier_adj_list

        # ====================
        trn_emb_src = ide_emb[trn_src_idxs]
        trn_emb_dst = ide_emb[trn_dst_idxs]
        # ==========
        val_emb_src = ide_emb[val_src_idxs]
        val_emb_dst = ide_emb[val_dst_idxs]
        # ==========
        tst_emb_src = ide_emb[tst_src_idxs]
        tst_emb_dst = ide_emb[tst_dst_idxs]
        del ide_emb

        # ====================
        auc_val, ap_val, auc_tst, ap_tst = LP_eva(
            trn_emb_src, trn_emb_dst, val_emb_src, val_emb_dst,
            tst_emb_src, tst_emb_dst, trn_gnd, val_gnd, tst_gnd)
        # ==========
        cat_auc_val.append(auc_val)
        cat_ap_val.append(ap_val)
        # ==========
        cat_auc_tst.append(auc_tst)
        cat_ap_tst.append(ap_tst)
        # ====================
        del trn_emb_src, trn_emb_dst, val_emb_src, val_emb_dst, tst_emb_src, tst_emb_dst, trn_gnd, val_gnd, tst_gnd
        torch.cuda.empty_cache()
        gc.collect()
    # ====================
    cat_auc_mean_val = np.mean(cat_auc_val)
    cat_auc_std_val = np.std(cat_auc_val)
    cat_ap_mean_val = np.mean(cat_ap_val)
    cat_ap_std_val = np.std(cat_ap_val)
    # ==========
    cat_auc_mean_tst = np.mean(cat_auc_tst)
    cat_auc_std_tst = np.std(cat_auc_tst)
    cat_ap_mean_tst = np.mean(cat_ap_tst)
    cat_ap_std_tst = np.std(cat_ap_tst)
    # ==========
    print('PI-HIST(A) %s d=%d L=%d EPS=%.1f; act=%s norm=%s n=%d'
          % (data_name, emb_dim, RW_len, eps, act, norm, num_RWs))
    print('VAL CAT AUC %.2f~(%.2f) AP %.2f~(%.2f)'
          % (cat_auc_mean_val*100, cat_auc_std_val*100,
             cat_ap_mean_val*100, cat_ap_std_val*100))
    print('TST CAT AUC %.2f~(%.2f) AP %.2f~(%.2f)'
          % (cat_auc_mean_tst*100, cat_auc_std_tst*100,
             cat_ap_mean_tst*100, cat_ap_std_tst*100))
