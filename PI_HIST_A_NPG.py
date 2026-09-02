# Node position classification and node identity clustering for PI-HIST (A)
# On datasets w/ node position ground-truth

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans

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
    print('NPC VAL-%.1f MICRO-F1 %.2f~(%.2f) MACRO-F1 %.2f~(%.2f)'
          % (val_ratio, mi_mean_val*100, mi_std_val*100, ma_mean_val*100, ma_std_val*100))
    # ====================
    mi_mean_tst = np.mean(micro_tst)
    mi_std_tst = np.std(micro_tst)
    ma_mean_tst = np.mean(macro_tst)
    ma_std_tst = np.std(macro_tst)
    print('NPC TST TRN-%.1f MICRO-F1 %.2f~(%.2f) MACRO-F1 %.2f~(%.2f)'
          % (trn_ratio, mi_mean_tst*100, mi_std_tst*100, ma_mean_tst*100, ma_std_tst*100))

def get_cond_mtc(edges, clus_res, num_clus):
    '''
    Function to get conductance metric w.r.t. a clustering result
    :param edges: edge list (undirected & 0-base node indices)
    :param clus_res: clustering result
    :param num_clus: number of clusters
    :return:
    '''
    # ====================
    cuts = [0.0 for _ in range(num_clus)]
    vols = [0.0 for _ in range(num_clus)]
    # ==========
    for (src, dst) in edges:
        # ==========
        src_lbl = clus_res[src]
        dst_lbl = clus_res[dst]
        # ==========
        vols[src_lbl] += 1.0
        vols[dst_lbl] += 1.0
        # ==========
        if src_lbl != dst_lbl:
            cuts[src_lbl] += 1.0
            cuts[dst_lbl] += 1.0
    # ==========
    cond = 0.0
    for c in range(num_clus):
        if vols[c] == 0:
            cond += 1.0
        else:
            cond += cuts[c] / vols[c]
    cond /= num_clus

    return cond

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str) # ppi, blogcatalog
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--L', type=int, default=5)
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

    tr_ratio = 0.2
    num_clus = 10

    # ====================
    # Load graph topology
    pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
    edges = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    # Load top-K similarity graph w.r.t. high-order deg feat
    pkl_file = open('data/%s_deg_sim_.pickle' % (data_name), 'rb')
    deg_sim_sp = pickle.load(pkl_file)
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
    num_clus = min(num_clus, num_clas)
    # ==========
    gnd = np.zeros((num_nodes, num_clas))
    for (node_idx, clus_idx) in gnd_sp:
        gnd[node_idx, clus_idx] = 1.0
    print('#NODES %d #EDGES %d #CLAS %d' % (num_nodes, num_edges, num_clas))

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

    # ====================
    print('PI-HIST(A) d=%d L=%d EPS=%f; act=%s, norm=%s, n=%d'
          % (emb_dim, RW_len, eps, act, norm, num_RWs))
    cond_list = []
    for rand_seed in range(10):
        kmeans = KMeans(n_clusters=num_clus, random_state=rand_seed).fit(ide_emb.astype(np.float64))
        clus_res = kmeans.labels_
        cond = get_cond_mtc(deg_sim_sp, clus_res, num_clus)
        cond_list.append(cond)
        #print('CLUS SEED=%d COND %.2f' % (rand_seed, cond*100))
    cond_mean = np.mean(cond_list)
    cond_std = np.std(cond_list)
    print('CLUS K=%d COND %.2f~(%.2f)' % (num_clus, cond_mean*100, cond_std*100))
    # ==========
    NPC_eva(ide_emb, gnd, trn_ratio=tr_ratio, n_splits=10, random_state=rand_seed_gbl)
    print()
