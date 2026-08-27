# Visualization for parameter analysis & ablation study of PI-HIST (A) on datasets w/ identity ground-truth
# Macro-F1 of node identity classification on validation set

import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import pickle

# ====================
data_name_map={
    'europe': 'Europe',
    'usa': 'USA',
    'actor': 'Actor',
    'film': 'Film',
    'ppi': 'PPI',
    'blogcatalog': 'BlogCatalog',
    'dblp': 'DBLP',
    'amazon': 'Amazon'
}

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str) # usa
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--norm', type=str, default='no') # no, l2, z
    parser.add_argument('--act', type=str, default='no') # no, tanh, sig, relu, exp
    args = parser.parse_args()

    # ====================
    data_name = args.data_name
    emb_dim = args.d
    norm = args.norm
    act = args.act

    # ====================
    RW_len_list = [5, 6, 7, 8, 9]  # 5 choices
    y_lbl = []
    for RW_len in RW_len_list:
        if RW_len == 5:
            y_lbl.append(r'$L$=%d' % (RW_len))
        else:
            y_lbl.append('%d' % (RW_len))
    # ==========
    eps_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 10 choices
    x_lbl = []
    for eps in eps_list:
        if eps == 0.0:
            x_lbl.append(r'$\varepsilon$=%.1f' % (eps))
        else:
            x_lbl.append('%.1f' % (eps))

    # ====================
    # Load save results (matrix form)
    pkl_file = open('res/PI-HIST(A)_NIGp_ma_%s_d=%d_act=%s_norm=%s.pickle'
                    % (data_name, emb_dim, act, norm), 'rb')
    res = pickle.load(pkl_file)
    pkl_file.close()
    res = res*100

    # =====================
    # Visualize avg correlation matrix
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(res,
                     cmap='YlGnBu',
                     annot=True, annot_kws={"size": 13},
                     cbar_kws={'shrink': 1.0},
                     vmin=30, vmax=65,
                     fmt='.2f',
                     linewidth=.5,
                     xticklabels=x_lbl,
                     yticklabels=y_lbl
                     )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15, rotation=0)

    # ====================
    if act == 'no':
        act_txt = '$\emptyset$'
    elif act == 'relu':
        act_txt = 'ReLU'
    elif act == 'sig':
        act_txt = 'sigmoid'
    else:
        act_txt = act
    # ==========
    if norm == 'no':
        norm_txt = '$\emptyset$'
    elif norm == 'l2':
        norm_txt = '$l_2$'
    elif norm == 'z':
        norm_txt = 'z-score'
    else:
        norm_txt = norm

    t = (r'PI-HIST (A) on %s,$\psi_{\rm{act}}$=%s,$\psi_{\rm{norm}}$=%s,Macro-F1$\uparrow$(%%)'
         % (data_name_map[data_name], act_txt, norm_txt))
    plt.title(t, fontsize=14, weight='bold', pad=15, loc='left')

    # ====================
    # Set font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # ====================
    plt.tight_layout()
    #plt.show()
    plt.savefig('vis/PI-HIST(A)_NIGp_%s_act=%s_norm=%s.svg' % (data_name, act, norm),
                format='svg',
                bbox_inches='tight',
                transparent=False)
    plt.close()
