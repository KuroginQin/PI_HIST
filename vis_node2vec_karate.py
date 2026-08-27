# Visualization for the reduced embeddings of node2vec on the Zachary's karate club network

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text

import numpy as np
import pickle

# Zachary's karate club graph
# https://scikit-network.readthedocs.io/en/latest/_modules/sknetwork/data/toy_graphs.html#karate_club

# ====================
gnd = [
1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
1, 1, 1, 1, 0, 0, 1, 1, 0, 1,
0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0]
# ==========
emb_dim = 16
num_nodes = len(gnd)

# ====================
# Load saved embedding
pkl_file = open('emb/node2vec_karate_d=%d_emb.pickle' % (emb_dim), 'rb')
pos_emb = pickle.load(pkl_file)
pkl_file.close()

# ====================
#for r in range(emb_dim):
#    z_mean = np.mean(pos_emb[:, r])
#    z_std = np.std(pos_emb[:, r])
#    emb[:, r] = (pos_emb[:, r] - z_mean) / z_std
#emb = preprocessing.normalize(pos_emb, axis=1)

# ====================
# Visualize emb
if emb_dim > 2:
    pos_emb_vis = TSNE(n_components=2, random_state=0).fit_transform(pos_emb)
else:
    pos_emb_vis = pos_emb
# ====================
for j in range(2):
    z_mean = np.mean(pos_emb_vis[:, j])
    z_std = np.std(pos_emb_vis[:, j])
    pos_emb_vis[:, j] = (pos_emb_vis[:, j] - z_mean) / z_std
#pos_emb_vis = preprocessing.normalize(pos_emb_vis, axis=1)

# ====================
plt.figure(figsize=(7, 6))
# ====================
# Draw scatter for (reduced) emb
for i in range(num_nodes):
    if gnd[i] == 0:
        plt.scatter(pos_emb_vis[i, 0], pos_emb_vis[i, 1], s=200, color='#F7A24F') # Officier's Community - orange
    else:
        plt.scatter(pos_emb_vis[i, 0], pos_emb_vis[i, 1], s=200, color='#79CAFB') # Mr. Hi's Community - blue
# ====================

txts = []
for i in range(num_nodes):
    if (i+1) == 1 or (i+1) == 34 or (i+1) == 9:
        txt = plt.text(pos_emb_vis[i, 0], pos_emb_vis[i, 1], r'$v_{%d}$' % (i+1), fontsize=18, fontweight='medium', color='black')
    else:
        txt = plt.text(pos_emb_vis[i, 0], pos_emb_vis[i, 1], '', fontsize=0, fontweight='medium', color='black')
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
plt.title('node2vec (Position Embedding)', fontsize=18, fontweight='bold')

# ====================
# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

plt.show()
plt.close()
