# Link prediction for PI-HIST (R&A) on all real graphs

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
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
    clas = LogisticRegression(random_state=rand_seed_gbl, solver='lbfgs', max_iter=5000)
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
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--d', type=int, default=64)
    # ==========
    parser.add_argument('--P_L', type=int, default=7)
    parser.add_argument('--P_eps', type=float, default=0.5)
    parser.add_argument('--P_norm', type=str, default='no') # no, l2, z
    parser.add_argument('--P_act', type=str, default='no') # no, tanh, sig, relu, exp# no, tanh, sig, relu, exp
    # ==========
    parser.add_argument('--I_L', type=int, default=7)
    parser.add_argument('--I_eps', type=float, default=0.5)
    parser.add_argument('--n', type=int, default=50000)
    parser.add_argument('--I_norm', type=str, default='no') # no, l2, z
    parser.add_argument('--I_act', type=str, default='no') # no, tanh, sig, relu, exp
    args = parser.parse_args()

    # ====================
    data_name = args.data_name
    emb_dim = args.d
    # ==========
    pos_RW_len = args.P_L
    pos_eps = args.P_eps
    pos_norm = args.P_norm
    pos_act = args.P_act
    # ==========
    ide_RW_len = args.I_L
    ide_eps = args.I_eps
    num_RWs = args.n
    ide_norm = args.I_norm
    ide_act = args.I_act
    # ==========
    alpha_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
    pkl_file = open('AW_hier_LP/AW_hier_bth_%s_L=%d_n=%d.pickle' % (data_name, ide_RW_len, num_RWs), 'rb')
    bth_idxs_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier_LP/AW_hier_src_%s_L=%d_n=%d.pickle' % (data_name, ide_RW_len, num_RWs), 'rb')
    src_idxs_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier_LP/AW_hier_dst_%s_L=%d_n=%d.pickle' % (data_name, ide_RW_len, num_RWs), 'rb')
    dst_idxs_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier_LP/AW_hier_vals_%s_L=%d_n=%d.pickle' % (data_name, ide_RW_len, num_RWs), 'rb')
    vals_list = pickle.load(pkl_file)
    pkl_file.close()

    # ====================
    for alpha in alpha_list:
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
            degs = [0.0 for _ in range(num_nodes)]
            for (src, dst) in trn_edges:
                degs[src] += 1.0
                degs[dst] += 1.0

            # ====================
            # PI-HIST (R) emb derivation
            sup_src_idxs, sup_dst_idxs, sup_vals = get_GNN_sup_sp(trn_edges, degs)
            idxs_tnr = torch.LongTensor([sup_src_idxs, sup_dst_idxs])
            vals_tnr = torch.FloatTensor(sup_vals)
            del sup_src_idxs, sup_dst_idxs, sup_vals
            sup_tnr = torch.sparse_coo_tensor(idxs_tnr, vals_tnr, size=[num_nodes, num_nodes]).to(device)
            # ==========
            init_pos_emb = get_rand_proj_mat(num_nodes, emb_dim, rand_seed=rand_seed_gbl)
            pos_emb_tnr = torch.FloatTensor(init_pos_emb).to(device)
            del init_pos_emb
            # ====================
            for _ in range(pos_RW_len):
                pos_emb_tnr = pos_eps*pos_emb_tnr + (1-pos_eps)*torch.spmm(sup_tnr, pos_emb_tnr)
                # ==========
                z_mean = torch.mean(pos_emb_tnr, dim=0).reshape((1, -1))
                z_std = torch.std(pos_emb_tnr, dim=0).reshape((1, -1))
                pos_emb_tnr = (pos_emb_tnr - z_mean) / z_std
            # ==========
            if pos_act == 'tanh':
                pos_emb_tnr = torch.tanh(pos_emb_tnr)
            elif pos_act == 'sig':
                pos_emb_tnr = torch.sigmoid(pos_emb_tnr)
            elif pos_act == 'relu':
                pos_emb_tnr = torch.relu(pos_emb_tnr)
            elif pos_act == 'exp':
                pos_emb_tnr = torch.exp(pos_emb_tnr)
            # ====================
            if pos_norm == 'l2':
                pos_emb_tnr = F.normalize(pos_emb_tnr, dim=1, p=2)
            elif pos_norm == 'z':
                z_mean = torch.mean(pos_emb_tnr, dim=0).reshape((1, -1))
                z_std = torch.std(pos_emb_tnr, dim=0).reshape((1, -1))
                pos_emb_tnr = (pos_emb_tnr - z_mean) / z_std
            # ====================
            if torch.cuda.is_available():
                pos_emb = pos_emb_tnr.cpu().data.numpy()
            else:
                pos_emb = pos_emb_tnr.data.numpy()
            del pos_emb_tnr, sup_tnr

            # ====================
            # PI-HIST (A) emb derivation
            adj_bth_idxs = bth_idxs_list[t]
            adj_src_idxs = src_idxs_list[t]
            adj_dst_idxs = dst_idxs_list[t]
            adj_vals = vals_list[t]
            # ====================
            hier_adj_list = []
            for r in range(ide_RW_len):
                idxs_tnr = torch.LongTensor([adj_bth_idxs[r],
                                             adj_src_idxs[r],
                                             adj_dst_idxs[r]])
                vals_tnr = torch.FloatTensor(adj_vals[r])
                hier_adj = torch.sparse_coo_tensor(idxs_tnr,
                                                   vals_tnr,
                                                   size=[num_nodes, ide_RW_len+1, ide_RW_len+1]).to(device)
                hier_adj_list.append(hier_adj)
            del adj_bth_idxs, adj_src_idxs, adj_dst_idxs, adj_vals
            # ====================
            init_ide_emb = get_rand_proj_mat(ide_RW_len + 1, emb_dim, rand_seed=rand_seed_gbl)
            ide_emb_tnr = torch.FloatTensor(init_ide_emb).to(device)
            del init_ide_emb
            # ====================
            for r in range(ide_RW_len-1, -1, -1):
                # ==========
                hier_adj_d = hier_adj_list[r].to_dense()
                ide_emb_tnr = ide_eps*ide_emb_tnr + (1-ide_eps)*torch.matmul(hier_adj_d, ide_emb_tnr)
                del hier_adj_d
                # ==========
                for k in range(r + 1):
                    for j in range(emb_dim):
                        z_mean = torch.mean(ide_emb_tnr[:, k, j])
                        z_std = torch.std(ide_emb_tnr[:, k, j])
                        if z_std > 0:
                            ide_emb_tnr[:, k, j] = (ide_emb_tnr[:, k, j] - z_mean) / z_std
            ide_emb_tnr = ide_emb_tnr[:, 0, :]
            if ide_act == 'tanh':
                ide_emb_tnr = torch.tanh(ide_emb_tnr)
            elif ide_act == 'sig':
                ide_emb_tnr = torch.sigmoid(ide_emb_tnr)
            elif ide_act == 'relu':
                ide_emb_tnr = torch.relu(ide_emb_tnr)
            elif ide_act == 'exp':
                ide_emb_tnr = torch.exp(ide_emb_tnr)
            # ====================
            #ide_emb_tnr = torch.nan_to_num(ide_emb_tnr)
            if ide_norm == 'l2':
                ide_emb_tnr = F.normalize(ide_emb_tnr, dim=1, p=2)
            elif ide_norm == 'z':
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
            emb = np.concatenate((alpha*pos_emb, (1-alpha)*ide_emb), axis=1)
            del pos_emb, ide_emb
            # ==========
            if data_name == 'actor' or data_name == 'amazon':
                for i in range(num_nodes):
                    crt_mean = np.mean(emb[i, :])
                    crt_std = np.std(emb[i, :])
                    if crt_std > 0.0:
                        emb[i, :] = (emb[i, :] - crt_mean) / crt_std
                #emb = np.nan_to_num(emb)
            else:
                emb = normalize(emb, 'l2')
                _, d = emb.shape
                for j in range(d):
                    crt_mean = np.mean(emb[:, j])
                    crt_std = np.std(emb[:, j])
                    if crt_std > 0.0:
                        emb[:, j] = (emb[:, j] - crt_mean) / crt_std
                #emb = np.nan_to_num(emb)

            # ====================
            trn_emb_src = emb[trn_src_idxs]
            trn_emb_dst = emb[trn_dst_idxs]
            # ==========
            val_emb_src = emb[val_src_idxs]
            val_emb_dst = emb[val_dst_idxs]
            # ==========
            tst_emb_src = emb[tst_src_idxs]
            tst_emb_dst = emb[tst_dst_idxs]
            del emb

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
        # ==========
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
        print('PI-HIST(R&A) d=%d P_L=%d P_eps=%.1f P_act=%s P_norm=%s '
              'I_L=%d I_eps=%.1f I_act=%s I_norm=%s alpha=%f'
              % (emb_dim, pos_RW_len, pos_eps, pos_act, pos_norm,
                 ide_RW_len, ide_eps, ide_act, ide_norm, alpha))
        print('VAL CAT AUC %.2f~(%.2f) AP %.2f~(%.2f)'
              % (cat_auc_mean_val*100, cat_auc_std_val*100,
                 cat_ap_mean_val*100, cat_ap_std_val*100))
        print('TST CAT AUC %.2f~(%.2f) AP %.2f~(%.2f)'
              % (cat_auc_mean_tst*100, cat_auc_std_tst*100,
                 cat_ap_mean_tst*100, cat_ap_std_tst*100))

