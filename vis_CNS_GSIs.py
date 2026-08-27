# Visualization for common neighbor signature (CNS)'s result of graph superfamily identification (GSI) on synthetic graphs

import numpy as np
import pickle

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns

# ====================
num_runs = 10 # Number of independent runs
num_graphs = 9 # Number of syn graph (in each run)
num_nodes = 1000

# ====================
pkl_file = open('emb/struc_div_stat_LFR.pickle', 'rb')
SD_stat_list = pickle.load(pkl_file)
pkl_file.close()

# ====================
avg_cor_mat = np.zeros((num_graphs, num_graphs))
for t in range(num_runs):
    # ====================
    SD_stat = np.array(SD_stat_list[t])
    # ==========
    SD_stat = normalize(SD_stat, 'l2', axis=0)
    #SD_stat = normalize(SD_stat, 'l2')
    # ==========
    #_, feat_dim = SD_stat.shape
    #for j in range(feat_dim):
    #    crt_mean = np.mean(SD_stat[:, j])
    #    crt_std = np.std(SD_stat[:, j])
    #    if crt_std > 0:
    #        SD_stat[:, j] = (SD_stat[:, j] - crt_mean) / crt_std
    #SD_stat = normalize(SD_stat, 'l2')

    #for i in range(num_syn_set):
    #    crt_mean = np.mean(SD_stat[i, :])
    #    crt_std = np.std(SD_stat[i, :])
    #    if crt_std > 0: SD_stat[i, :] = (SD_stat[i, :] - crt_mean) / crt_std
    #SD_stat = normalize(SD_stat, 'l2')

    # ==========
    #cor_mat = np.dot(SD_stat, SD_stat.transpose())
    cor_mat = np.corrcoef(SD_stat)
    #print(cor_mat)
    #print()
    # ==========
    avg_cor_mat += cor_mat
# ==========
avg_cor_mat = avg_cor_mat / num_runs
#print(avg_cor_mat)

# =====================
plt.figure(figsize=(8, 6))
ax = sns.heatmap(avg_cor_mat,
                 cmap='RdBu_r',
                 annot=True, annot_kws={"size": 16},
                 square=True,
                 cbar_kws={'shrink': 1.0},
                 vmin=0.0, vmax=1.0,
                 fmt='.2f',
                 linewidth=.5,
                 xticklabels=[r'$G_{%d}$' % (i) for i in range(1, num_graphs+1)],
                 yticklabels=[r'$G_{%d}$' % (i) for i in range(1, num_graphs+1)]
                 )
ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)

plt.title('CNS', fontsize=16, weight='bold', pad=15, loc='left')

# ====================
# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# ====================
plt.tight_layout()
plt.show()
plt.close()
