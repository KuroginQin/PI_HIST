# Node identity classification and community detection for PI-HIST (R)
# On datasets w/ node identity ground-truth

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F

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
    print('NIC VAL-%.1f MICRO-F1 %.2f~(%.2f) MACRO-F1 %.2f~(%.2f)'
          % (val_ratio, mi_mean_val*100, mi_std_val*100, ma_mean_val*100, ma_std_val*100))
    # ====================
    mi_mean_tst = np.mean(micro_tst)
    mi_std_tst = np.std(micro_tst)
    ma_mean_tst = np.mean(macro_tst)
    ma_std_tst = np.std(macro_tst)
    print('NIC TST TRN-%.1f MICRO-F1 %.2f~(%.2f) MACRO-F1 %.2f~(%.2f)'
          % (trn_ratio, mi_mean_tst*100, mi_std_tst*100, ma_mean_tst*100, ma_std_tst*100))

def get_mod_mtc(edges, clus_res, num_clus):
    '''
    Function to get modularity metric w.r.t. a clustering result
    :param edges: edge list (undirected & 0-base node indices)
    :param clus_res: clustering result
    :param num_clus: number of clusters
    :return:
    '''
    # ====================
    mod_mtc = 0.0
    # ==========
    num_edges = len(edges)*2 # Number of edges
    clus_in_edges = [0 for _ in range(num_clus)] # Number of intra-cluster edges w.r.t. each cluster
    clus_edges = [0 for _ in range(num_clus)] # Number of induced edges w.r.t. each cluster
    # ==========
    for (src, dst) in edges:
        # ==========
        src_lbl = clus_res[src]
        dst_lbl = clus_res[dst]
        # ==========
        if src_lbl==dst_lbl:
            clus_in_edges[src_lbl] += 2
        clus_edges[src_lbl] += 1
        clus_edges[dst_lbl] += 1
    # ==========
    for lbl in range(num_clus):
        L = clus_in_edges[lbl]/num_edges
        R = clus_edges[lbl]/num_edges
        mod_mtc += (L - R*R)

    return mod_mtc

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
    parser.add_argument('--data_name', type=str) # europe, usa, actor, film
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--norm', type=str, default='no') # no, l2, z
    parser.add_argument('--act', type=str, default='no') # no, tanh, sig, relu, exp
    parser.add_argument('--phi_flag', type=bool, default=True)
    args = parser.parse_args()

    # ====================
    data_name = args.data_name
    emb_dim = args.d
    RW_len = args.L
    eps = args.eps
    norm = args.norm
    act = args.act
    phi_flag = args.phi_flag

    tr_ratio = 0.2
    num_clus = 10

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
    num_clus = min(num_clus, num_clas)
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
    # ==========
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
        #pos_emb_tnr = torch.nan_to_num(pos_emb_tnr)
    # ==========
    if torch.cuda.is_available():
        pos_emb = pos_emb_tnr.cpu().data.numpy()
    else:
        pos_emb = pos_emb_tnr.data.numpy()

    # ====================
    print('PI-HIST(R) d=%d L=%d EPS=%f; act=%s, norm=%s'
          % (emb_dim, RW_len, eps, act, norm))
    mod_list = []
    for rand_seed in range(10):
        kmeans = KMeans(n_clusters=num_clus, random_state=rand_seed).fit(pos_emb.astype(np.float64))
        clus_res = kmeans.labels_
        mod = get_mod_mtc(edges, clus_res, num_clus)
        mod_list.append(mod)
        #print('CLUS SEED=%d MOD %.2f' % (rand_seed, mod*100))
    mod_mean = np.mean(mod_list)
    mod_std = np.std(mod_list)
    print('CLUS K=%d MOD %.2f~(%.2f)' % (num_clus, mod_mean*100, mod_std*100))
    NIC_eva(pos_emb, gnd, trn_ratio=tr_ratio, n_splits=10, random_state=rand_seed_gbl)
    print()
