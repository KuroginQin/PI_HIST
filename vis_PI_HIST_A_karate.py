# Visualization for the reduced embeddings of PI-HIST (A) on the Zachary's karate club network

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text

import numpy as np
import pickle

# Zachary's karate club graph
# https://scikit-network.readthedocs.io/en/latest/_modules/sknetwork/data/toy_graphs.html#karate_club

# ====================
emb_dim = 16

# ====================
# Load saved embedding
pkl_file = open('emb/PI-HIST(A)_karate_d=%d_emb.pickle' % (emb_dim), 'rb')
ide_emb = pickle.load(pkl_file)
pkl_file.close()
# ==========
num_nodes, _ = ide_emb.shape

# ====================
#for r in range(emb_dim):
#    z_mean = np.mean(ide_emb[:, r])
#    z_std = np.std(ide_emb[:, r])
#    emb[:, r] = (ide_emb[:, r] - z_mean) / z_std
#emb = preprocessing.normalize(ide_emb, axis=1)

# ====================
# Visualize emb
if emb_dim > 2:
    ide_emb_vis = TSNE(n_components=2, random_state=0).fit_transform(ide_emb)
else:
    ide_emb_vis = ide_emb
# ====================
for j in range(2):
    z_mean = np.mean(ide_emb_vis[:, j])
    z_std = np.std(ide_emb_vis[:, j])
    ide_emb_vis[:, j] = (ide_emb_vis[:, j] - z_mean) / z_std
#ide_emb_vis = preprocessing.normalize(ide_emb_vis, axis=1)

# ====================
plt.figure(figsize=(7, 6))
# ====================
# Draw scatter for (reduced) emb
for i in range(num_nodes):
    if (i + 1) == 1 or (i + 1) == 34:
        plt.scatter(ide_emb_vis[i, 0], ide_emb_vis[i, 1], s=100, color='white', linewidths=2,
                    edgecolors='#AA77E9')  # leaders - purple
    elif (i + 1) == 2 or (i + 1) == 3 or (i + 1) == 4 or (i + 1) == 33:
        plt.scatter(ide_emb_vis[i, 0], ide_emb_vis[i, 1], s=100, color='white', linewidths=2,
                    edgecolors='#4EA660')  # strong supporters - green
    else:
        plt.scatter(ide_emb_vis[i, 0], ide_emb_vis[i, 1], s=100, color='white', linewidths=2,
                    edgecolors='#E95351')  # ordinary members - red
# ====================
txts = []
for i in range(num_nodes):
    if (i+1) == 1 or (i+1) == 34 or (i+1) == 2 or (i+1) == 3 or (i+1) == 4 or (i+1) == 33:
        txt = plt.text(ide_emb_vis[i, 0], ide_emb_vis[i, 1], r'$v_{%d}$' % (i+1), fontsize=18, fontweight='medium', color='black')
    else:
        txt = plt.text(ide_emb_vis[i, 0], ide_emb_vis[i, 1], '', fontsize=18, fontweight='medium', color='black')
    txts.append(txt)
# ==========
adjust_text(
    txts,
    arrowprops=dict(arrowstyle='-', color='gray', lw=1),
    force_static=5.0,
    force_text=5.0,
    expand_text=(5, 5),
    lim=5000,
    precision=0.01,
)

# ====================
# Set figure title
plt.title('PI-HIST (A)', fontsize=18, fontweight='bold')

# ====================
# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

plt.show()
plt.close()