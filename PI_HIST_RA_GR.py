# Graph reconstruction for PI-HIST (R&A) on all real graphs

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

def GR_eva(emb, pred_src_list, pred_dst_list, pred_gnd_list, num_runs, rand_state=0):
    # ====================
    auc_list = []
    ap_list = []
    for t in range(num_runs):
        # ====================
        pred_src_idxs = pred_src_list[t]
        pred_dst_idxs = pred_dst_list[t]
        pred_gnd = pred_gnd_list[t]
        # ==========
        emb_src = emb[pred_src_idxs]
        emb_dst = emb[pred_dst_idxs]
        # ==========
        cat_input = np.concatenate((emb_src, emb_dst), axis=1)
        clas = LogisticRegression(random_state=rand_state, solver='lbfgs', max_iter=5000)
        clas.fit(cat_input, pred_gnd)
        probs = clas.predict_proba(cat_input)
        probs = probs[:, 1]
        del cat_input
        # ==========
        auc = roc_auc_score(pred_gnd, probs)
        ap = average_precision_score(pred_gnd, probs)  # Average Precision
        # ==========
        auc_list.append(auc)
        ap_list.append(ap)
    # ==========
    auc_mean = np.mean(auc_list)
    auc_std = np.std(auc_list)
    ap_mean = np.mean(ap_list)
    ap_std = np.std(ap_list)
    print('BIN-CLAS CAT AUC %.2f~(%.2f) AP %.2f~(%.2f)'
          % (auc_mean*100, auc_std*100, ap_mean*100, ap_std*100))

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
    parser.add_argument('--data_name', type=str) # europe, usa, actor, film, ppi, blogcatalog, dblp, amazon
    parser.add_argument('--d', type=int, default=64)
    # ==========
    parser.add_argument('--P_L', type=int, default=7)
    parser.add_argument('--P_eps', type=float, default=0.5)
    parser.add_argument('--P_norm', type=str, default='no') # no, l2, z
    parser.add_argument('--P_act', type=str, default='no') # no, tanh, sig, relu, exp
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

    # ====================
    pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
    edges = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    num_nodes = np.max(edges) + 1
    num_edges = len(edges)
    print('#NODES %d #EDGES %d' % (num_nodes, num_edges))

    # ====================
    pkl_file = open('data_LP/GR_%s_pos.pickle' % (data_name), 'rb')
    pos_pairs_list = pickle.load(pkl_file, encoding='bytes')
    pkl_file.close()
    # ==========
    pkl_file = open('data_LP/GR_%s_neg.pickle' % (data_name), 'rb')
    neg_pairs_list = pickle.load(pkl_file, encoding='bytes')
    pkl_file.close()
    # ==========
    num_runs = len(pos_pairs_list)

    # ====================
    pkl_file = open('AW_hier/AW_hier_bth_%s_L=%d_n=%d.pickle'
                    % (data_name, ide_RW_len, num_RWs), 'rb')
    adj_bth_idxs = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier/AW_hier_src_%s_L=%d_n=%d.pickle'
                    % (data_name, ide_RW_len, num_RWs), 'rb')
    adj_src_idxs = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier/AW_hier_dst_%s_L=%d_n=%d.pickle'
                    % (data_name, ide_RW_len, num_RWs), 'rb')
    adj_dst_idxs = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW_hier/AW_hier_vals_%s_L=%d_n=%d.pickle'
                    % (data_name, ide_RW_len, num_RWs), 'rb')
    adj_vals = pickle.load(pkl_file)
    pkl_file.close()

    # ====================
    degs = [0.0 for _ in range(num_nodes)]
    for (src, dst) in edges:
        degs[src] += 1.0
        degs[dst] += 1.0

    # ====================
    pred_src_list = []
    pred_dst_list = []
    pred_gnd_list = []
    for t in range(num_runs):
        # ====================
        pos_pairs = pos_pairs_list[t]
        neg_pairs = neg_pairs_list[t]
        # ==========
        pred_src_idxs = []
        pred_dst_idxs = []
        pred_gnd = []
        # ==========
        for (src, dst) in pos_pairs:
            pred_src_idxs.append(src)
            pred_dst_idxs.append(dst)
            pred_gnd.append(1.0)
        # ==========
        for (src, dst) in neg_pairs:
            pred_src_idxs.append(src)
            pred_dst_idxs.append(dst)
            pred_gnd.append(0.0)
        # ==========
        pred_src_list.append(pred_src_idxs)
        pred_dst_list.append(pred_dst_idxs)
        pred_gnd_list.append(pred_gnd)
    del pos_pairs_list, neg_pairs_list

    # ====================
    sup_src_idxs, sup_dst_idxs, sup_vals = get_GNN_sup_sp(edges, degs)
    idxs_tnr = torch.LongTensor([sup_src_idxs, sup_dst_idxs])
    vals_tnr = torch.FloatTensor(sup_vals)
    sup_tnr = torch.sparse_coo_tensor(idxs_tnr, vals_tnr, size=[num_nodes, num_nodes]).to(device)
    # ==========
    init_pos_emb = get_rand_proj_mat(num_nodes, emb_dim, rand_seed=rand_seed_gbl)
    pos_emb_tnr = torch.FloatTensor(init_pos_emb).to(device)
    del sup_src_idxs, sup_dst_idxs, idxs_tnr, init_pos_emb

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
        del idxs_tnr, vals_tnr
    # ====================
    init_ide_emb = get_rand_proj_mat(ide_RW_len+1, emb_dim, rand_seed=rand_seed_gbl)
    ide_emb_tnr = torch.FloatTensor(init_ide_emb).to(device)
    del init_ide_emb

    # ====================
    for r in range(ide_RW_len-1, -1, -1):
        # ==========
        hier_adj_d = hier_adj_list[r].to_dense()
        ide_emb_tnr = ide_eps*ide_emb_tnr + (1-ide_eps)*torch.matmul(hier_adj_d, ide_emb_tnr)
        del hier_adj_d
        # ==========
        for k in range(r+1):
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

    # ====================
    if torch.cuda.is_available():
        ide_emb = ide_emb_tnr.cpu().data.numpy()
    else:
        ide_emb = ide_emb_tnr.data.numpy()
    del ide_emb_tnr, hier_adj_list

    # ====================
    for alpha in alpha_list:
        # ====================
        print('PI-HIST(R&A) d=%d P_L=%d P_eps=%.1f P_act=%s P_norm=%s '
              'I_L=%d I_eps=%.1f I_act=%s I_norm=%s alpha=%f'
              % (emb_dim, pos_RW_len, pos_eps, pos_act, pos_norm,
                 ide_RW_len, ide_eps, ide_act, ide_norm, alpha))
        emb = np.concatenate((alpha*pos_emb, (1-alpha)*ide_emb), axis=1)
        # ==========
        for i in range(num_nodes):
            crt_mean = np.mean(emb[i, :])
            crt_std = np.std(emb[i, :])
            if crt_std > 0.0:
                emb[i, :] = (emb[i, :] - crt_mean) / crt_std
        GR_eva(emb, pred_src_list, pred_dst_list, pred_gnd_list, num_runs, rand_state=0)
        del emb
        gc.collect()
    print()
