# Graph reconstruction for PI-HIST (A) on all real graphs

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
        clas = LogisticRegression(random_state=rand_state, solver='lbfgs')
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
        del idxs_tnr, vals_tnr
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

    # ====================
    if torch.cuda.is_available():
        ide_emb = ide_emb_tnr.cpu().data.numpy()
    else:
        ide_emb = ide_emb_tnr.data.numpy()
    del ide_emb_tnr, hier_adj_list

    # ====================
    print('PI-HIST(A) d=%d L=%d EPS=%f; act=%s, norm=%s, n=%d'
          % (emb_dim, RW_len, eps, act, norm, num_RWs))
    GR_eva(ide_emb, pred_src_list, pred_dst_list, pred_gnd_list, num_runs, rand_state=rand_seed_gbl)
    print()
