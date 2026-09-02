# Efficiency evaluation for the AW-induced hierarchical structure extraction of PI-HIST (A)

import torch
from torch_cluster import random_walk
from numba import cuda, types

import gc
import argparse
import numpy as np
import random
import timeit
import pickle

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@cuda.jit
def RW2AW_kernel(RW_array, AW_array, walk_length):
    '''
    Kernel function to map a batch of RWs to corresponding AWs
    :param RW_array: RWs input
    :param AW_array: AWs output
    :param walk_length: RW/AW length
    :return:
    '''
    idx = cuda.grid(1)
    if idx < RW_array.shape[0]:
        walk = RW_array[idx]
        local_dict = cuda.local.array(shape=(10), dtype=np.int32)  # Assume the max index < 10
        count = 0
        for i in range(walk_length):
            found = False
            for j in range(count):
                if walk[i] == local_dict[j]:
                    AW_array[idx, i] = j
                    found = True
                    break
            if not found:
                local_dict[count] = walk[i]
                AW_array[idx, i] = count
                count += 1

def RW2AW_batch(RW_array):
    '''
    Numba batch function to map sampled RWs to corresponidng AWs
    :param RW_array: RWs input
    :return: AWs output
    '''
    # ====================
    batch_size, walk_length = RW_array.shape
    AW_array = torch.zeros_like(RW_array).to(RW_array.device)
    threads_per_block = 256
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
    rw_array_gpu = cuda.to_device(RW_array)
    aw_array_gpu = cuda.to_device(AW_array)
    # ====================
    RW2AW_kernel[blocks_per_grid, threads_per_block](rw_array_gpu, aw_array_gpu, walk_length)
    del rw_array_gpu

    return aw_array_gpu

def pre_DP_table(L):
    '''
    Function to precompute dynamic programming (DP) table for AW index mapping
    :param L: RW_len + 1
    :return:
    '''
    # ====================
    # Initialization
    g = [[0] * (L+1) for _ in range(L+1)]
    # ==========
    for max_val in range(L+1):
        g[0][max_val] = 1

    # ====================
    # State Transition
    for r_len in range(1, L+1):
        for max_val in range(L+1):
            term1 = (max_val+1) * g[r_len-1][max_val]
            term2 = g[r_len-1][max_val+1] if max_val + 1 <= L else 0
            g[r_len][max_val] = term1 + term2

    return g

@cuda.jit
def AW_idx_kernel(AWs_batch, DP_table, AW_idxs):
    '''
    Kernel function to map a batch of AWs to corresponding indices
    :param AWs_batch: AW batch input
    :param DP_table: pre-compute DP table
    :param AW_idxs: AW indices output
    :return:
    '''
    # ====================
    idx = cuda.grid(1)
    batch_size, L = AWs_batch.shape

    # ====================
    if idx < batch_size:
        # ====================
        AW = AWs_batch[idx] # AW to be mapped
        # ====================
        # DP algorithm for AW index mapping
        L = 0
        for i in range(AW.shape[0]):
            if AW[i] == -1:
                L = i
                break
        if L == 0:
            L = AW.shape[0]
        # ==========
        crt_max_idx = 0 # Current max index
        r_len = L - 1
        acc_idx = 0 # Accumulated index
        # ==========
        for i in range(1, L):
            x = AW[i]
            # ==========
            if x < 0 or x > crt_max_idx+1:
                AW_idxs[idx] = -1
                return
            # ==========
            table_idx = (r_len-1)*(L+1)+crt_max_idx
            if x <= crt_max_idx:
                acc_idx += x*DP_table[table_idx]
            else: # x == crt_max_idx + 1
                acc_idx += (crt_max_idx+1) * DP_table[table_idx]
                crt_max_idx += 1
            # ==========
            r_len -= 1
        # ==========
        AW_idxs[idx] = acc_idx

def AW_idx_batch(AWs, d_DP_table):
    '''
    Numba batch function to map processed AWs to corresponding indices
    :param AWs: AWs to be mapped
    :param DP_table: pre-computed DP table
    :return: AW indices output
    '''
    # ====================
    AW_idxs = cuda.device_array(AWs.shape[0], dtype=np.int32)
    # ====================
    threadsperblock = 256
    blockspergrid = (AWs.shape[0] + threadsperblock - 1) // threadsperblock

    AW_idx_kernel[blockspergrid, threadsperblock](AWs, d_DP_table, AW_idxs)

    return AW_idxs

def get_AW_hier(rows, cols, vals, RW_len, AW_list, device='cuda'):
    '''
    Function to get AW-induced hierarchical structures based on the estimated AW stat
    :param rows: row indices of C (i.e., sparase matrix form of AW stat)
    :param cols: column indices of C
    :param vals: corresponding values of C
    :param RW_len: RW/AW length
    :param AW_list: list of all length-L AWs
    :param device:
    :return: sparse tensor about AW-induced hierarchical structures for all nodes
    '''
    # ====================
    AW_list_tensor = torch.tensor(AW_list, dtype=torch.long, device=device)
    # ==========
    total_entries = len(rows)
    chunk_size = 500000000 # Adjust according to GPU memory
    num_chunks = (total_entries+chunk_size-1) // chunk_size

    # ====================
    # Map the 3-tuple (batch_idx, src, dst) in the tensor form of AW-induced hier struc via hash
    all_edge_info = [[] for _ in range(RW_len)] # List to store 3-tuple
    all_weights = [[] for _ in range(RW_len)] # List to store corresponding hier struc weights
    # ==========
    for chunk_idx in range(num_chunks):
        # ====================
        # Get AW stat (i.e., row indices, column indices, and values) to be processed in current chunk
        start_idx = chunk_idx*chunk_size
        end_idx = min((chunk_idx+1)*chunk_size, total_entries)
        # ==========
        chunk_rows = rows[start_idx:end_idx]
        chunk_cols = cols[start_idx:end_idx]
        chunk_vals = vals[start_idx:end_idx]
        # ==========
        chunk_AW_seqs = AW_list_tensor[chunk_cols] # Get AWs w.r.t. given AW indices (i.e., column indices of C)

        # ====================
        for p in range(RW_len): # For each element in AW
            # ====================
            src = chunk_AW_seqs[:, p]
            dst = chunk_AW_seqs[:, p+1]
            # ====================
            # Get each 3-tuple (batch_idx, src, dst) map it to another unique index via large prime hashing
            batch_idx = chunk_rows
            prime = 1000003
            edge_hash = batch_idx*prime*prime + src*prime + dst

            # ====================
            # Merge all associated weights using the efficient unique operation
            unique_hashes, inverse_indices = torch.unique(edge_hash, return_inverse=True)
            edge_weights = torch.zeros(unique_hashes.shape[0], device=device)
            edge_weights.scatter_add_(0, inverse_indices, chunk_vals)
            # ==========
            mask = edge_weights > 1e-8 # Filter too small weights
            if mask.any():
                # ==========
                valid_hashes = unique_hashes[mask]
                valid_weights = edge_weights[mask]
                # ==========
                # Recovery the hashed 3-tuple
                valid_dst = valid_hashes % prime
                valid_src = (valid_hashes // prime) % prime
                valid_batch = valid_hashes // (prime * prime)
                # ==========
                # Store the recovered results
                all_edge_info[p].append(torch.stack([valid_batch, valid_src, valid_dst], dim=1))
                all_weights[p].append(valid_weights)
        # ====================
        # Regularly clear the GPU cache
        if device == 'cuda' and chunk_idx % 5 == 0:
            torch.cuda.empty_cache()

    # ====================
    adj_bth_idxs = [] # Node index
    adj_src_idxs = [] # Unit index in layer (p)
    adj_dst_idxs = [] # Unit index in layer (p+1)
    adj_vals = [] # Weights of the hier struc
    # ==========
    for p in range(RW_len):
        # ====================
        # For the case w/o edges
        if not all_edge_info[p]:
            adj_bth_idxs.append([])
            adj_src_idxs.append([])
            adj_dst_idxs.append([])
            adj_vals.append([])
            continue
        # ====================
        # Merge results from all chunks
        combined_edges = torch.cat(all_edge_info[p], dim=0) # [total_edges, 3]
        combined_weights = torch.cat(all_weights[p], dim=0) # [total_edges]
        # ==========
        # Merge possibly duplicate weights using large prime hash and unique operation
        prime = 1000003
        global_hash = combined_edges[:, 0]*prime*prime + combined_edges[:, 1]*prime + combined_edges[:, 2]
        unique_global_hashes, inverse_indices = torch.unique(global_hash, return_inverse=True)
        global_weights = torch.zeros(unique_global_hashes.shape[0], device=device)
        global_weights.scatter_add_(0, inverse_indices, combined_weights)

        # ====================
        mask = global_weights > 1e-8 # Mask to skip too small weights
        if mask.any():
            # ====================
            valid_hashes = unique_global_hashes[mask]
            valid_weights = global_weights[mask]
            # ==========
            # Recovery 3-tupes
            valid_dst = valid_hashes % prime
            valid_src = (valid_hashes // prime) % prime
            valid_batch = valid_hashes // (prime * prime)
            # ====================
            # Move results to CPU
            adj_bth_idxs.append(valid_batch.cpu().numpy().astype(int).tolist())
            adj_src_idxs.append(valid_src.cpu().numpy().astype(int).tolist())
            adj_dst_idxs.append(valid_dst.cpu().numpy().astype(int).tolist())
            adj_vals.append(valid_weights.cpu().numpy().astype(float).tolist())
        else:
            adj_bth_idxs.append([])
            adj_src_idxs.append([])
            adj_dst_idxs.append([])
            adj_vals.append([])
        # ==========
        # Clear GPU memory
        if device == 'cuda':
            torch.cuda.empty_cache()

    return adj_bth_idxs, adj_src_idxs, adj_dst_idxs, adj_vals

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str) # europe, usa, actor, film, ppi, blogcatalog, dblp, amazon
    parser.add_argument('--L', type=int, default=7)
    parser.add_argument('--n', type=int, default=50000) # Number of RWs per node
    parser.add_argument('--np', type=int, default=50000) # Number of RWs per node in each iteration
    parser.add_argument('--nr', type=int, default=10) # Number of independent runs
    args = parser.parse_args()

    # ====================
    data_name = args.data_name
    RW_len = args.L # RW length
    num_RWs = args.n # Number of RWs per node
    num_RWs_iter = args.np # Number of RWs per node in each iteration
    num_runs = args.nr # Number of independent runs
    warmups = 5

    # ====================
    if num_RWs < num_RWs_iter:
        num_iter = 1
        num_RWs_iter = num_RWs
    else:
        num_iter = num_RWs // num_RWs_iter # Number of iterations/batches

    # ====================
    # Load graph topology
    pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
    edges = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    pkl_file = open('AW/AWs_L=%d.pickle' % (RW_len), 'rb')
    AW_list = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    num_edges = len(edges)
    num_nodes = np.max(np.max(edges)) + 1
    num_AWs = len(AW_list)
    print('#NODES %d #EDGS %d L=%d #AWs %d' % (num_nodes, num_edges, RW_len, num_AWs))

    # ====================
    # Pre-process data for RW sampling
    src_idxs = []
    dst_idxs = []
    for (src, dst) in edges:
        # ==========
        src_idxs.append(src)
        dst_idxs.append(dst)
        # ==========
        src_idxs.append(dst)
        dst_idxs.append(src)
    del edges
    src_tnr_ = torch.LongTensor(src_idxs)
    dst_tnr_ = torch.LongTensor(dst_idxs)
    # ===========
    start = []
    for i in range(num_nodes):
        for _ in range(num_RWs_iter):
            start.append(i)
    start_ = torch.tensor(start)
    torch.cuda.empty_cache()

    # ====================
    # Pre-compute dynamic programming (DP) table for AW index mapping
    DP_table = pre_DP_table(len(AW_list[0]))
    DP_table = np.array(DP_table)
    DP_table_flat = DP_table.ravel().astype(np.int32)
    # ==========
    row_idxs_ = torch.arange(num_nodes).unsqueeze(1).expand(-1, num_RWs_iter)
    flat_row_idxs_ = row_idxs_.reshape(-1)

    # ====================
    for iter in range(warmups+num_runs):
        # ====================
        if iter == warmups:
            print()

        # ====================
        torch.cuda.empty_cache()
        gc.collect()
        # ====================
        torch.cuda.synchronize()

        # ====================
        RW_time_gbl = 0.0
        AW_map_time_gbl = 0.0
        AW_idx_time_gbl = 0.0
        cnt_time_gbl = 0.0
        # ==========
        ttl_IO_comm_time = 0.0

        # ====================
        # Move pre-processed data to GPU
        time_s = timeit.default_timer()
        d_DP_table = cuda.to_device(DP_table_flat)
        src_tnr = src_tnr_.to(device)
        dst_tnr = dst_tnr_.to(device)
        start = start_.to(device)
        flat_row_idxs = flat_row_idxs_.to(device)
        time_e = timeit.default_timer()
        ttl_IO_comm_time += time_e - time_s

        # ====================
        rand_seed = 0  # Random seed for RW sampling
        # ==========
        idxs_gbl = []
        vals_gbl = []
        for iter in range(num_iter):
            # ====================
            set_seed(rand_seed)
            rand_seed += 1

            # ====================
            # Phase 1: RW sampling
            time_s = timeit.default_timer()
            RWs_tnr = random_walk(src_tnr, dst_tnr, start, RW_len)
            time_e = timeit.default_timer()
            RW_time = time_e - time_s
            RW_time_gbl += RW_time

            # ====================
            # Phase 2: AW mapping
            time_s = timeit.default_timer()
            AWs_tnr = RW2AW_batch(RWs_tnr)
            time_e = timeit.default_timer()
            AW_map_time = time_e - time_s
            AW_map_time_gbl += AW_map_time
            del RWs_tnr

            # ====================
            # Phase 3: AW indexing
            time_s = timeit.default_timer()
            AW_idxs = AW_idx_batch(AWs_tnr, d_DP_table)
            del AWs_tnr
            AW_idxs = torch.as_tensor(AW_idxs, device=device)
            time_e = timeit.default_timer()
            AW_idx_time = time_e - time_s
            AW_idx_time_gbl += AW_idx_time

            # ====================
            # Phase 4: AW counting
            time_s = timeit.default_timer()
            comb = torch.stack((flat_row_idxs, AW_idxs), dim=1)
            del AW_idxs
            unq_comb, cnts = torch.unique(comb, return_counts=True, dim=0)
            del comb
            time_e = timeit.default_timer()
            cnt_time = time_e - time_s
            cnt_time_gbl += cnt_time

            # ====================
            #print('ITER %d/%d: RW %f AW-MAP %f AW-IDX %f CNT %f'
            #      % (iter+1, num_iter, RW_time, AW_map_time, AW_idx_time, cnt_time))

            # ====================
            time_s = timeit.default_timer()
            idxs_gbl.append(unq_comb.cpu())
            vals_gbl.append(cnts.cpu())
            # ==========
            time_e = timeit.default_timer()
            ttl_IO_comm_time += time_e - time_s

            # ====================
            del unq_comb, cnts
            torch.cuda.empty_cache()
        # ====================
        del src_tnr, dst_tnr, start, flat_row_idxs, d_DP_table
        torch.cuda.empty_cache()
        gc.collect()

        # ====================
        # Phase 5: statistic gathering
        time_s = timeit.default_timer()
        # ==========
        idxs_gbl = torch.cat(idxs_gbl, dim=0)
        vals_gbl = torch.cat(vals_gbl)
        # ==========
        AW_stat_sp_gbl = torch.sparse_coo_tensor(
            idxs_gbl.t(),
            vals_gbl,
            size=(num_nodes, num_AWs),
            device=device
        )
        # ==========
        AW_stat_sp_gbl = AW_stat_sp_gbl.coalesce()
        AW_stat_sp_gbl = AW_stat_sp_gbl / num_RWs
        time_e = timeit.default_timer()
        gat_time = time_e - time_s

        # ====================
        idxs = AW_stat_sp_gbl.indices()
        vals = AW_stat_sp_gbl.values()
        del AW_stat_sp_gbl
        torch.cuda.empty_cache()
        gc.collect()

        # ====================
        # Phase 6: hierarchical structure extraction
        time_s = timeit.default_timer()
        adj_bth_idxs, adj_src_idxs, adj_dst_idxs, adj_vals = get_AW_hier(idxs[0, :], idxs[1, :], vals, RW_len, AW_list)
        time_e = timeit.default_timer()
        hier_time = time_e - time_s
        del adj_bth_idxs, adj_src_idxs, adj_dst_idxs, adj_vals, idxs, vals
        torch.cuda.empty_cache()
        gc.collect()

        # ====================
        ttl_comp_time = (RW_time_gbl + AW_map_time_gbl + AW_idx_time_gbl + cnt_time_gbl
                         + gat_time + hier_time)
        print('AW TTL %.4f RW %.4f AW-MAP %.4f AW-IDX %.4f CNT %.4f GAT %.4f HIER %.4f'
              % (ttl_comp_time,
                 RW_time_gbl, AW_map_time_gbl, AW_idx_time_gbl, cnt_time_gbl,
                 gat_time, hier_time))
