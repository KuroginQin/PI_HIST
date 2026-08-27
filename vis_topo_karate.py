# Visualization for the example topology of the Zachary's karate club network

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Zachary's karate club network
# https://scikit-network.readthedocs.io/en/latest/_modules/sknetwork/data/toy_graphs.html#karate_club

# ====================
src_idxs = [
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
2, 2, 3, 3, 3, 4, 4, 5, 5, 5,
6, 8, 8, 8, 9, 13, 14, 14, 15, 15,
18, 18, 19, 20, 20, 22, 22, 23, 23, 23,
23, 23, 24, 24, 24, 25, 26, 26, 27, 28,
28, 29, 29, 30, 30, 31, 31, 32]
# ==========
dst_idxs = [
1, 2, 3, 4, 5, 6, 7, 8, 10, 11,
12, 13, 17, 19, 21, 31, 2, 3, 7, 13,
17, 19, 21, 30, 3, 7, 8, 9, 13, 27,
28, 32, 7, 12, 13, 6, 10, 6, 10, 16,
16, 30, 32, 33, 33, 33, 32, 33, 32, 33,
32, 33, 33, 32, 33, 32, 33, 25, 27, 29,
32, 33, 25, 27, 31, 31, 29, 33, 33, 31,
33, 32, 33, 32, 33, 32, 33, 33]
# ==========
gnd = [
1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
1, 1, 1, 1, 0, 0, 1, 1, 0, 1,
0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0]
# ==========
num_nodes = len(gnd)
num_edges = len(src_idxs)
num_clus = np.max(gnd) + 1

# ====================
edges = []
for i in range(num_edges):
    src_idx = src_idxs[i] + 1
    dst_idx = dst_idxs[i] + 1
    # ==========
    edges.append((src_idx, dst_idx))
edges = sorted(edges)

# ====================
G = nx.Graph()
G.add_edges_from(edges)
pos = nx.kamada_kawai_layout(G)

# ====================
# Set node colors and node-edge colors w.r.t. node pos & ide
node_clrs = []
edge_clrs = []
for node in G.nodes:
    # ====================
    lbl = gnd[node-1]
    if lbl == 0:
        node_clrs.append('#F7A24F') # Officier's Community - orange
    else:
        node_clrs.append('#79CAFB') # Mr. Hi's Community - blue
    # ====================
    if node == 1 or node==34:
        edge_clrs.append('#AA77E9') # leaders - purple
    elif node==2 or node==3 or node==4 or node==33:
        edge_clrs.append('#4EA660') # strong supporters - green
    else:
        edge_clrs.append('#E95351') # ordinary members - red
# ====================
node_lbls = {node: r'$v_{%d}$' % (node) for node in G.nodes}

# ====================
plt.figure(figsize=(7, 6))
# ====================
# Draw graph topology
nx.draw(G, pos, node_size=900,
        labels=node_lbls, with_labels=True,
        node_color=node_clrs, edgecolors=edge_clrs, linewidths=3.0, font_size=16, font_color='black', width=2)
# ====================
# Draw legend
class_info = {
    'Leaders': '#AA77E9', # purple circles
    'Strong Supporters': '#4EA660', # green circles
    'Ordinary Members': '#E95351' # red circles
}
for label, color in class_info.items():
    plt.scatter([], [], c='none', edgecolors=color, linewidths=2, label=label)
# ==========
club_color = {
    'Mr. Hi\'s': '#79CAFB', # blue nodes
    'Officers\'': '#F7A24F' # orange nodes
}
for label, color in club_color.items():
    plt.scatter([], [], c=color, label=label, s=100)
plt.legend(title="", loc="upper left", fontsize=15.5, ncol=2, frameon=False)

# ====================
#eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
#for i in range(num_nodes):
#    c = eigenvector_centrality[i+1]
#    print('NODE %d CEN %f' % (i+1, c))

# ====================
# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# ====================
plt.axis('off')
plt.tight_layout()
plt.show()
plt.close()