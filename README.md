# Topology-induced Operators Reveal Complementary Graph Representations without Training

### Abstract
Graph representation learning has largely focused on designing increasingly sophisticated models to transform graph topology into vector representations, or embeddings. However, the extent to which embedding quality depends on model learning, rather than on the underlying topological transformations, remains unclear. Here, we show that informative embeddings can be derived without complicated model design and gradient-based training. Propagating random features through implicit hierarchical structures induced by random walks and anonymous walks yields embeddings that capture node proximity and structural role, respectively. These two training-free embeddings preserve complementary aspects of graph organization and perform competitively with classic and recent methods across various node-, edge-, and graph-level tasks. They often require substantially less computation, resulting in a favorable quality–efficiency trade-off. Combining the two types of embeddings further improves inference quality of some tasks compared with using either embedding type alone. Our results suggest that informative graph embeddings can arise from carefully chosen topological transformations before any learned operation is applied.

### Citing
TBD

If you have any questions regarding this repository, you can contact the author via [mengqin_az@foxmail.com].

### Requirements
- numpy
- scipy
- scikit-learn
- networkx
- pytorch
- pytorch_cluster
- numba
- pandas
- matplotlib
- seaborn
- adjustText

To install the experiment environment:
```bash
conda create -n pi_hist python=3.11 -y
conda activate pi_hist

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

conda install -c nvidia cuda-toolkit=12.8 -y
conda install -c conda-forge numba -y
conda install -c conda-forge libstdcxx-ng gcc gxx_linux-64

pip install "numpy<2.3" numba==0.61.2 scikit-learn==1.6.1 networkx==3.4.2 \
    scipy==1.15.2 matplotlib==3.10.0 seaborn==0.13.2 pandas==2.2.3 adjustText==1.3.0
```

### Usage
***
### Visualizing Main Results
The pre-computed results for the visualization have been placed in ```./res/```. The pre-computed embeddings and features for the visualization of graph-level case studies have been placed in ```./emb/```. Run the following commands to visualize results of experimental evaluations and case studies.

#### **Proof-of-concept** (Fig. 1)
To visualize the example **topology** of **the Zachary's karate club network** (Fig. 1a)
```bash
python vis_topo_karate.py
```

To visualize the reduced embeddings of **node2vec** and **struc2vec** on the **Zachary's karate club network** (Fig. 1d-e)
```bash
python vis_node2vec_karate.py
python vis_struc2vec_karate.py
```

To visualize the reduced embeddings of **PI-HIST (R)** and **PI-HIST (A)** on **the Zachary's karate club network** (Fig. 1f-g)
```bash
python vis_PI_HIST_R_karate.py
python vis_PI_HIST_A_karate.py
```

#### **Node-level Evaluations** (Fig. 2 and Supplementary Fig. S1)
To visualize results of **node identity classification** (macro- and micro-F1) and **community detection** (modularity) on **Europe**,  **USA**, **Actor**, and **Film** (datasets with node identity ground-truth):
```bash
python vis_NIG.py --data_name europe --qlt_lbl macro
python vis_NIG.py --data_name europe --qlt_lbl micro
python vis_NIG.py --data_name europe --qlt_lbl mod

python vis_NIG.py --data_name usa --qlt_lbl macro
python vis_NIG.py --data_name usa --qlt_lbl micro
python vis_NIG.py --data_name usa --qlt_lbl mod

python vis_NIG.py --data_name actor --qlt_lbl macro
python vis_NIG.py --data_name actor --qlt_lbl micro
python vis_NIG.py --data_name actor --qlt_lbl mod

python vis_NIG.py --data_name film --qlt_lbl macro
python vis_NIG.py --data_name film --qlt_lbl micro
python vis_NIG.py --data_name film --qlt_lbl mod
```

To visualize results of **node position classification** (macro- and micro-F1) and **node identity clustering** (conductance) on **PPI** and **BlogCatalog** (small-scale datasets with node position ground-truth):
```bash
python vis_NPG.py --data_name ppi --qlt_lbl macro
python vis_NPG.py --data_name ppi --qlt_lbl micro
python vis_NPG.py --data_name ppi --qlt_lbl cond

python vis_NPG.py --data_name blogcatalog --qlt_lbl macro
python vis_NPG.py --data_name blogcatalog --qlt_lbl micro
python vis_NPG.py --data_name blogcatalog --qlt_lbl cond
```

To visualize results of **node position classification** (macro- and micro-F1) and **node identity clustering** (conductance) on **DBLP** and **Amazon** (large-scale datasets with node position ground-truth):
```bash
python vis_NPG_.py --data_name dblp --qlt_lbl macro
python vis_NPG_.py --data_name dblp --qlt_lbl micro
python vis_NPG_.py --data_name dblp --qlt_lbl cond

python vis_NPG_.py --data_name amazon --qlt_lbl macro
python vis_NPG_.py --data_name amazon --qlt_lbl micro
python vis_NPG_.py --data_name amazon --qlt_lbl cond
```

#### **Edge-level Evaluations** (Fig. 3 ans Supplementary Fig. S2)
To visualize results of **link prediction** (AUC) and **graph reconstruction** (AUC) on **Europe**, **USA**, **PPI**, **Actor**, **BlogCatalog**, and **Film** (small-scale real-world graphs):
```bash
python vis_LPGR.py --data_name europe --task lp
python vis_LPGR.py --data_name europe --task gr

python vis_LPGR.py --data_name usa --task lp
python vis_LPGR.py --data_name usa --task gr

python vis_LPGR.py --data_name ppi --task lp
python vis_LPGR.py --data_name ppi --task gr

python vis_LPGR.py --data_name actor --task lp
python vis_LPGR.py --data_name actor --task gr

python vis_LPGR.py --data_name blogcatalog --task lp
python vis_LPGR.py --data_name blogcatalog --task gr

python vis_LPGR.py --data_name film --task lp
python vis_LPGR.py --data_name film --task gr
```

To visualize results of **link prediction** (AUC) and **graph reconstruction** (AUC) on **DBLP** and **Amazon** (large-scale real-world graphs):
```bash
python vis_LPGR_.py --data_name dblp --task lp
python vis_LPGR_.py --data_name dblp --task gr

python vis_LPGR_.py --data_name amazon --task lp
python vis_LPGR_.py --data_name amazon --task gr
```

#### **Graph-level Case studies** (Fig. 4, Supplementary Fig. S3, and Supplementary Fig. S4)
To visualize the **graph superfamily identification** results of **PI-HIST (R)** and **PI-HIST (A)** on **real-world graphs** (Fig. 4c):
```bash
python vis_PI_HIST_R_GSIr.py --d 64 --L 5 --eps 0.9 --act relu --norm no
python vis_PI_HIST_A_GSIr.py --d 64 --L 7 --eps 0.8 --act relu --norm no
```

To visualize the **graph superfamily identification** result of **PI-HIST (R&A)** with $\alpha=0.9$ on **real-world graphs** (Fig. 4c):
```bash
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 0.9
```

To visualize the **graph superfamily identification** results of **PI-HIST (R&A)** w.r.t. the remaining settings of $\alpha$ on **real-world graphs** (Supplementary Fig. S3a):
```bash
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 0.0
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 0.1
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 0.2
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 0.3
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 0.4
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 0.5
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 0.6
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 0.7
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 0.8
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 1.0
```

To visualize the **graph superfamily identification** results of **BoD** (bag of degrees), **BoHD** (bag of high-order degrees), **SSP** (subgraph significance profile), and **CNS** (common neighbor signature) on **real-world graphs** (Supplementary Fig. S4b):
```bash
python vis_BoD_GSIr.py
python vis_BoHD_GSIr.py
python vis_SSP_GSIr.py
python vis_CNS_GSIr.py
```

To visualize the **graph superfamily identification** results of **PI-HIST (R)** and **PI-HIST (A)** on **synthetic graphs** (Fig. 4d)
```bash
python vis_PI_HIST_R_GSIs.py --d 64 --L 5 --eps 0.8 --act relu --norm no
python vis_PI_HIST_A_GSIs.py --d 64 --L 6 --eps 0.5 --act relu --norm no
```

To visualize the **graph superfamily identification** result of **PI-HIST (R&A)** with $\alpha=0.6$ on **synthetic graphs** (Fig. 4d):
```bash
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 0.6
```

To visualize the **graph superfamily identification** results of **PI-HIST (R&A)** w.r.t. the remaining settings of $\alpha$ on **real-world graphs** (Supplementary Fig. S3b):
```bash
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 0.0
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 0.1
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 0.2
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 0.3
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 0.4
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 0.5
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 0.7
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 0.8
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 0.9
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 1.0
```

To visualize the **graph superfamily identification** results of **BoD**, **BoHD**, **SSP**, and **CNS** on **synthetic graphs** (Supplementary Fig. S4b):
```bash
python vis_BoD_GSIs.py
python vis_BoHD_GSIs.py
python vis_SSP_GSIs.py
python vis_CNS_GSIs.py
```

#### **Ablation Study and Parameter Analysis** (Supplementary Fig. S5， S6， S7, and S8)
To visualize the **ablation study and parameter analysis** results of **PI-HIST (R)** on **PPI** with node position ground-truth (Supplementary Fig. S5):
```bash
python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act no --norm no
python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act no --norm l2

python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act relu --norm no
python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act relu --norm l2
python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act relu --norm z

python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act tanh --norm no
python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act tanh --norm l2
python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act tanh --norm z

python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act sig --norm no
python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act sig --norm l2
python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act sig --norm z

python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act exp --norm no
python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act exp --norm l2
python vis_PI_HIST_R_NPGp.py --data_name ppi --d 256 --act exp --norm z
```

To visualize the **ablation study and parameter analysis** results of **PI-HIST (R)** on **USA** with node position ground-truth (Supplementary Fig. S6):
```bash
python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act no --norm no
python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act no --norm l2

python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act relu --norm no
python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act relu --norm l2
python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act relu --norm z

python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act tanh --norm no
python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act tanh --norm l2
python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act tanh --norm z

python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act sig --norm no
python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act sig --norm l2
python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act sig --norm z

python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act exp --norm no
python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act exp --norm l2
python vis_PI_HIST_R_NIGp.py --data_name usa --d 64 --act exp --norm z
```

To visualize the **ablation study and parameter analysis** results of **PI-HIST (A)** on **USA** with node identity ground-truth (Supplementary Fig. S7):
```bash
python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act no --norm no
python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act no --norm l2

python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act relu --norm no
python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act relu --norm l2
python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act relu --norm z

python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act tanh --norm no
python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act tanh --norm l2
python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act tanh --norm z

python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act sig --norm no
python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act sig --norm l2
python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act sig --norm z

python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act exp --norm no
python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act exp --norm l2
python vis_PI_HIST_A_NIGp.py --data_name usa --d 64 --act exp --norm z
```

To visualize the **ablation study and parameter analysis** results of **PI-HIST (A)** on **PPI** with node identity ground-truth (Supplementary Fig. S8):
```bash
python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act no --norm no
python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act no --norm l2

python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act relu --norm no
python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act relu --norm l2
python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act relu --norm z

python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act tanh --norm no
python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act tanh --norm l2
python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act tanh --norm z

python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act sig --norm no
python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act sig --norm l2
python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act sig --norm z

python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act exp --norm no
python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act exp --norm l2
python vis_PI_HIST_A_NPGp.py --data_name ppi --d 256 --act exp --norm z
```


***
### Preparing Data
The pre-processed real-world and synthetic graph datasets for node- and graph-level tasks have been placed in ```./data/```. The generation process of synthetic graphs can be checked in ```./LFR_syn_gen.py```.

Download and extract the [pre-processed real-world datasets](https://drive.google.com/file/d/1upLsGSdGFL9eDTdgC5iPYJKBgJ7XrOFu/view?usp=sharing) (~117MB) for edge-level tasks with split for multiple runs. Place the extracted files in ```./data_LP/```. Details about the data split can be checked in ```./LP_data_prep.py``` and ```./GR_data_prep.py```.

Download the [pre-computed AW-induced hierarchical structures](https://drive.google.com/file/d/1NU5SPUfoIAxNmoadroDMPpeH1TFVAx8m/view?usp=sharing) (~951MB) for PI-HIST (A) on node- and graph-level tasks. Place the extracted files in ```./AW_hier/```.

Download the [pre-computed AW-induced hierarchical structures](https://drive.google.com/file/d/1kUSFObCxoW8QYVfXmEwS3sGDAUrcsAFh/view?usp=sharing) (~416MB) for PI-HIST (A) on edge-level tasks. Place the extracted files in ```./AW_hier_LP/```.


***
### Reproducing Main Results

#### **Node-level Evaluations**
To conduct **node position classification** and **node identity clustering** for **PI-HIST (R)** on the second type of datasets with position ground-truth (i.e., **PPI**, **BlogCatalog**, **DBLP**, and **Amazon**):
```bash
python PI_HIST_R_NPG.py --data_name ppi --d 256 --L 10 --eps 0.2 --act tanh --norm no
python PI_HIST_R_NPG.py --data_name blogcatalog --d 512 --L 6 --eps 0.1 --act no --norm no
python PI_HIST_R_NPG_.py --data_name dblp --d 256 --L 20 --eps 0.0 --act tanh --norm z
python PI_HIST_R_NPG_.py --data_name amazon --d 128 --L 20 --eps 0.1 --act tanh --norm no
```

To conduct **node identity classification** and **community detection** for **PI-HIST (R)** on the first type of datasets with identity ground-truth (i.e., **Europe**, **USA**, **Actor**, and **Film**):
```bash
python PI_HIST_R_NIG.py --data_name europe --d 64 --L 5 --eps 0.3 --act no --norm l2
python PI_HIST_R_NIG.py --data_name usa --d 64 --L 10 --eps 0.7 --act no --norm l2
python PI_HIST_R_NIG.py --data_name actor --d 256 --L 15 --eps 0.3 --act no --norm l2
python PI_HIST_R_NIG.py --data_name film --d 256 --L 12 --eps 0.1 --act relu --norm l2
```

To conduct **node identity classification** and **community detection** for **PI-HIST (A)** on the first type of datasets with identity ground-truth (i.e., **Europe**, **USA**, **Actor**, and **Film**):
```bash
python PI_HIST_A_NIG.py --data_name europe --d 64 --L 5 --eps 0.9 --act no --norm no --n 50000
python PI_HIST_A_NIG.py --data_name usa --d 64 --L 5 --eps 0.3 --act tanh --norm z --n 50000
python PI_HIST_A_NIG.py --data_name actor --d 256 --L 5 --eps 0.9 --act relu --norm no --n 50000
python PI_HIST_A_NIG.py --data_name film --d 256 --L 7 --eps 0.4 --act relu --norm z --n 50000
```

To conduct **node position classification** and **node identity clustering** for **PI-HIST (A)** on the second type of datasets with position ground-truth (i.e., **PPI**, **BlogCatalog**, **DBLP**, and **Amazon**):
```bash
python PI_HIST_A_NPG.py --data_name ppi --d 256 --L 8 --eps 0.6 --act sig --norm l2 --n 50000
python PI_HIST_A_NPG.py --data_name blogcatalog --d 512 --L 8 --eps 0.7 --act relu --norm no --n 50000
python PI_HIST_A_NPG_.py --data_name dblp --d 256 --L 8 --eps 0.0 --act relu --norm z --n 10000
python PI_HIST_A_NPG_.py --data_name amazon --d 128 --L 7 --eps 0.1 --act relu --norm no --n 10000
```

Note that the pre-compute AW-induced hierarchical structures for **PI-HIST (A)** have been placed in ```AW_hier```. To pre-compute such hierarchical structures from scratch:
```bash
python PI_HIST_AW_hier.py --data_name europe --L 5 --n 50000 --np 50000
python PI_HIST_AW_hier.py --data_name usa --L 5 --n 50000 --np 50000
python PI_HIST_AW_hier.py --data_name actor --L 5 --n 50000 --np 50000
python PI_HIST_AW_hier.py --data_name film --L 7 --n 50000 --np 10000
python PI_HIST_AW_hier.py --data_name ppi --L 8 --n 50000 --np 50000
python PI_HIST_AW_hier.py --data_name blogcatalog --L 8 --n 50000 --np 25000
python PI_HIST_AW_hier.py --data_name dblp --L 8 --n 10000 --np 500
python PI_HIST_AW_hier.py --data_name amazon --L 7 --n 10000 --np 500
```

To conduct **efficiency analysis** for **PI-HIST (R)** on all real-world graph datasets:
```bash
python PI_HIST_R_T.py --data_name europe --d 64 --L 5 --eps 0.3 --act no --norm l2
python PI_HIST_R_T.py --data_name usa --d 64 --L 10 --eps 0.7 --act no --norm l2
python PI_HIST_R_T.py --data_name actor --d 256 --L 15 --eps 0.3 --act no --norm l2
python PI_HIST_R_T.py --data_name film --d 256 --L 12 --eps 0.1 --act relu --norm l2
python PI_HIST_R_T.py --data_name ppi --d 256 --L 10 --eps 0.2 --act tanh --norm no
python PI_HIST_R_T.py --data_name blogcatalog --d 512 --L 6 --eps 0.1 --act no --norm no
python PI_HIST_R_T.py --data_name dblp --d 256 --L 20 --eps 0.0 --act tanh --norm z
python PI_HIST_R_T.py --data_name amazon --d 128 --L 20 --eps 0.1 --act tanh --norm no
```

To conduct **efficiency analysis** for **AW-induced hierarchial structure extraction** of **PI-HIST (A)** on all real-world graph dataset:
```bash
python PI_HIST_AW_T.py --data_name europe --L 5 --n 50000 --np 50000
python PI_HIST_AW_T.py --data_name usa --L 5 --n 50000 --np 50000
python PI_HIST_AW_T.py --data_name actor --L 5 --n 50000 --np 50000
python PI_HIST_AW_T.py --data_name film --L 7 --n 50000 --np 10000
python PI_HIST_AW_T.py --data_name ppi --L 8 --n 50000 --np 50000
python PI_HIST_AW_T.py --data_name blogcatalog --L 8 --n 50000 --np 25000
python PI_HIST_AW_T.py --data_name dblp --L 8 --n 10000 --np 500
python PI_HIST_AW_T.py --data_name amazon --L 7 --n 10000 --np 500
```

To conduct **efficiency analysis** for **training-free feedforward propagation** of **PI-HIST (A)** on all real-world graph datasets:
```bash
python PI_HIST_A_T.py --data_name europe --d 64 --L 5 --eps 0.9 --act no --norm no --n 50000
python PI_HIST_A_T.py --data_name usa --d 64 --L 5 --eps 0.3 --act tanh --norm z --n 50000
python PI_HIST_A_T.py --data_name actor --d 256 --L 5 --eps 0.9 --act relu --norm no --n 50000
python PI_HIST_A_T.py --data_name film --d 256 --L 7 --eps 0.4 --act relu --norm z --n 50000
python PI_HIST_A_T.py --data_name ppi --d 256 --L 8 --eps 0.6 --act sig --norm l2 --n 50000
python PI_HIST_A_T.py --data_name blogcatalog --d 512 --L 8 --eps 0.7 --act relu --norm no --n 50000
python PI_HIST_A_T.py --data_name dblp --d 256 --L 8 --eps 0.0 --act relu --norm z --n 10000
python PI_HIST_A_T.py --data_name amazon --d 256 --L 7 --eps 0.0 --act relu --norm no --n 10000
```

#### **Edge-level Evaluations**
To conduct **link prediction** for **PI-HIST (R)** on all the real graph datasets:
```bash
python PI_HIST_R_LP.py --data_name europe --d 64 --L 5 --eps 0.0 --act relu --norm z
python PI_HIST_R_LP.py --data_name usa --d 64 --L 5 --eps 0.5 --act relu --norm z
python PI_HIST_R_LP.py --data_name ppi --d 256 --L 5 --eps 0.6 --act relu --norm z
python PI_HIST_R_LP.py --data_name actor --d 256 --L 5 --eps 0.7 --act relu --norm z
python PI_HIST_R_LP.py --data_name blogcatalog --d 512 --L 5 --eps 0.0 --act relu --norm z
python PI_HIST_R_LP.py --data_name film --d 256 --L 5 --eps 0.6 --act relu --norm z
python PI_HIST_R_LP.py --data_name dblp --d 256 --L 5 --eps 0.7 --act exp --norm z
python PI_HIST_R_LP.py --data_name amazon --d 128 --L 5 --eps 0.6 --act exp --norm z
```
To conduct **link prediction** for **PI-HIST (A)** on all the real graph datasets:
```bash
python PI_HIST_A_LP.py --data_name europe --d 64 --L 7 --eps 0.2 --act sig --norm z --n 50000
python PI_HIST_A_LP.py --data_name usa --d 64 --L 8 --eps 0.9 --act relu --norm z --n 50000
python PI_HIST_A_LP.py --data_name ppi --d 256 --L 8 --eps 0.5 --act tanh --norm z --n 50000
python PI_HIST_A_LP.py --data_name actor --d 256 --L 6 --eps 0.3 --act relu --norm z --n 50000
python PI_HIST_A_LP.py --data_name blogcatalog --d 512 --L 5 --eps 0.3 --act relu --norm z --n 50000
python PI_HIST_A_LP.py --data_name film --d 256 --L 6 --eps 0.4 --act relu --norm z --n 50000
python PI_HIST_A_LP.py --data_name dblp --d 256 --L 5 --eps 0.8 --act relu --norm z --n 10000
python PI_HIST_A_LP.py --data_name amazon --d 128 --L 5 --eps 0.6 --act exp --norm z --n 10000
```
To conduct **link prediction** for **PI-HIST (R&A)** on all the real graph datasets:
```bash
python PI_HIST_RA_LP.py --data_name europe --d 64 --P_L 5 --P_eps 0.0 --P_act relu --P_norm z --I_L 7 --I_eps 0.2 --I_act sig --I_norm z --n 50000
python PI_HIST_RA_LP.py --data_name usa --d 64 --P_L 5 --P_eps 0.5 --P_act relu --P_norm z --I_L 8 --I_eps 0.9 --I_act relu --I_norm z --n 50000
python PI_HIST_RA_LP.py --data_name ppi --d 256 --P_L 5 --P_eps 0.6 --P_act relu --P_norm z --I_L 8 --I_eps 0.5 --I_act tanh --I_norm z --n 50000
python PI_HIST_RA_LP.py --data_name actor --d 256 --P_L 5 --P_eps 0.7 --P_act relu --P_norm z --I_L 6 --I_eps 0.3 --I_act relu --I_norm z --n 50000
python PI_HIST_RA_LP.py --data_name blogcatalog --d 512 --P_L 5 --P_eps 0.0 --P_act relu --P_norm z --I_L 5 --I_eps 0.3 --I_act relu --I_norm z --n 50000
python PI_HIST_RA_LP.py --data_name film --d 256 --P_L 5 --P_eps 0.6 --P_act relu --P_norm z --I_L 6 --I_eps 0.4 --I_act relu --I_norm z --n 50000
python PI_HIST_RA_LP.py --data_name dblp --d 256 --P_L 5 --P_eps 0.7 --P_act exp --P_norm z --I_L 5 --I_eps 0.8 --I_act relu --I_norm z --n 10000
python PI_HIST_RA_LP.py --data_name amazon --d 128 --P_L 5 --P_eps 0.6 --P_act exp --P_norm z --I_L 5 --I_eps 0.6 --I_act exp --I_norm z --n 10000
```

To conduct **graph reconstruction** for **PI-HIST (R)** on all the real graph datasets:
```bash
python PI_HIST_R_GR.py --data_name europe --d 64 --L 5 --eps 0.0 --act relu --norm z
python PI_HIST_R_GR.py --data_name usa --d 64 --L 5 --eps 0.5 --act relu --norm z
python PI_HIST_R_GR.py --data_name ppi --d 256 --L 7 --eps 0.7 --act exp --norm z
python PI_HIST_R_GR.py --data_name actor --d 256 --L 6 --eps 0.8 --act exp --norm z
python PI_HIST_R_GR.py --data_name blogcatalog --d 512 --L 5 --eps 0.0 --act tanh --norm z
python PI_HIST_R_GR.py --data_name film --d 256 --L 5 --eps 0.8 --act exp --norm z
python PI_HIST_R_GR.py --data_name dblp --d 256 --L 5 --eps 0.5 --act relu --norm z
python PI_HIST_R_GR.py --data_name amazon --d 128 --L 6 --eps 0.0 --act tanh --norm z
```
To conduct **graph reconstruction** for **PI-HIST (A)** on all the real graph datasets:
```bash
python PI_HIST_A_GR.py --data_name europe --d 64 --L 8 --eps 0.2 --act relu --norm z --n 50000
python PI_HIST_A_GR.py --data_name usa --d 64 --L 6 --eps 0.3 --act relu --norm z --n 50000
python PI_HIST_A_GR.py --data_name ppi --d 256 --L 8 --eps 0.3 --act relu --norm z --n 50000
python PI_HIST_A_GR.py --data_name actor --d 256 --L 7 --eps 0.3 --act relu --norm z --n 50000
python PI_HIST_A_GR.py --data_name blogcatalog --d 512 --L 5 --eps 0.3 --act relu --norm z --n 50000
python PI_HIST_A_GR.py --data_name film --d 256 --L 5 --eps 0.9 --act relu --norm z --n 50000
python PI_HIST_A_GR.py --data_name dblp --d 256 --L 6 --eps 0.2 --act relu --norm z --n 10000
python PI_HIST_A_GR.py --data_name amazon --d 128 --L 7 --eps 0.2 --act relu --norm z --n 10000
```
To conduct **graph reconstruction** for **PI-HIST (R&A)** on all the real graph datasets:
```bash
python PI_HIST_RA_GR.py --data_name europe --d 64 --P_L 5 --P_eps 0.0 --P_act relu --P_norm z --I_L 8 --I_eps 0.2 --I_act relu --I_norm z --n 50000
python PI_HIST_RA_GR.py --data_name usa --d 64 --P_L 5 --P_eps 0.5 --P_act relu --P_norm z --I_L 6 --I_eps 0.3 --I_act relu --I_norm z --n 50000
python PI_HIST_RA_GR.py --data_name ppi --d 256 --P_L 7 --P_eps 0.7 --P_act exp --P_norm z --I_L 8 --I_eps 0.3 --I_act relu --I_norm z --n 50000
python PI_HIST_RA_GR.py --data_name actor --d 256 --P_L 6 --P_eps 0.8 --P_act exp --P_norm z --I_L 7 --I_eps 0.3 --I_act relu --I_norm z --n 50000
python PI_HIST_RA_GR.py --data_name blogcatalog --d 512 --P_L 5 --P_eps 0.0 --P_act tanh --P_norm z --I_L 5 --I_eps 0.3 --I_act relu --I_norm z --n 50000
python PI_HIST_RA_GR.py --data_name film --d 256 --P_L 5 --P_eps 0.8 --P_act exp --P_norm z --I_L 5 --I_eps 0.9 --I_act relu --I_norm z --n 50000
python PI_HIST_RA_GR.py --data_name dblp --d 256 --P_L 5 --P_eps 0.5 --P_act relu --P_norm z --I_L 6 --I_eps 0.2 --I_act relu --I_norm z --n 10000
python PI_HIST_RA_GR.py --data_name amazon --d 128 --P_L 6 --P_eps 0.0 --P_act tanh --P_norm z --I_L 7 --I_eps 0.2 --I_act relu --I_norm z --n 10000
```

Note that the pre-compute AW-induced hierarchical structures for **PI-HIST (A)** have been placed in ```AW_hier_LP```. To pre-compute such hierarchical structures from scratch:
```bash
python PI_HIST_AW_hier_LP.py --data_name europe --L 7 --n 50000 --np 50000
python PI_HIST_AW_hier_LP.py --data_name usa --L 8 --n 50000 --np 50000
python PI_HIST_AW_hier_LP.py --data_name ppi --L 8 --n 50000 --np 50000
python PI_HIST_AW_hier_LP.py --data_name actor --L 6 --n 50000 --np 50000
python PI_HIST_AW_hier_LP.py --data_name blogcatalog --L 5 --n 50000 --np 50000
python PI_HIST_AW_hier_LP.py --data_name film --L 6 --n 50000 --np 10000
python PI_HIST_AW_hier_LP.py --data_name dblp --L 5 --n 10000 --np 1000
python PI_HIST_AW_hier_LP.py --data_name amazon --L 5 --n 10000 --np 1000
```

#### **Graph-level Case Studies**
To pre-compute **PI-HIST (R)** and **PI-HIST (A) embeddings** for **graph superfamily identification** on **real graphs**:
```bash
python PI_HIST_R_GSIr.py --d 64 --L 5 --eps 0.9 --act relu --norm no
python PI_HIST_A_GSIr.py --d 64 --L 7 --eps 0.8 --act relu --norm no
```
To check (i.e., visualize) results of **PI-HIST (R)**, **PI-HIST (A)**, and **PI-HIST (R&A)** on **real graphs**:
```bash
python vis_PI_HIST_R_GSIr.py --d 64 --L 5 --eps 0.9 --act relu --norm no
python vis_PI_HIST_A_GSIr.py --d 64 --L 7 --eps 0.8 --act relu --norm no
python vis_PI_HIST_RA_GSIr.py --d 64 --P_L 5 --P_eps 0.9 --P_act relu --P_norm no --I_L 7 --I_eps 0.8 --I_act relu --I_norm no --alpha 0.9
```
Note that the pre-computed **AW-induced hierarchical structures** for **PI-HIST (A)** have been placed in ```./AW_hier/```. To pre-compute such **hierarchical structures** from scratch:
```bash
python PI_HIST_AW_hier.py --data_name europe --L 7 --n 50000 --np 50000
python PI_HIST_AW_hier.py --data_name usa --L 7 --n 50000 --np 50000
python PI_HIST_AW_hier.py --data_name actor --L 7 --n 50000 --np 50000
python PI_HIST_AW_hier.py --data_name film --L 7 --n 50000 --np 10000
python PI_HIST_AW_hier.py --data_name ppi --L 7 --n 50000 --np 50000
python PI_HIST_AW_hier.py --data_name blogcatalog --L 7 --n 50000 --np 25000
python PI_HIST_AW_hier.py --data_name dblp --L 7 --n 20000 --np 500
python PI_HIST_AW_hier.py --data_name amazon --L 7 --n 20000 --np 500
```

To pre-compute **PI-HIST (R)** and **PI-HIST (A) embeddings** for **graph superfamily identification** on **synthetic graphs**:
```bash
python PI_HIST_R_GSIs.py --d 64 --L 5 --eps 0.8 --act relu --norm no
python PI_HIST_A_GSIs.py --d 64 --L 6 --eps 0.5 --act relu --norm no
```
To check (i.e., visualize) results of **PI-HIST (R)**, **PI-HIST (A)**, and **PI-HIST (R&A)** on **synthetic graphs**:
```bash
python vis_PI_HIST_R_GSIs.py --d 64 --L 5 --eps 0.8 --act relu --norm no
python vis_PI_HIST_A_GSIs.py --d 64 --L 6 --eps 0.5 --act relu --norm no
python vis_PI_HIST_RA_GSIs.py --d 64 --P_L 5 --P_eps 0.8 --P_act relu --P_norm no --I_L 6 --I_eps 0.5 --I_act relu --I_norm no --alpha 0.6
```
Note that the pre-computed **AW-induced hierarchical structures** for **PI-HIST (A)** have been placed in ```./AW_hier/```. To pre-compute such **hierarchical structures** from scratch:
```bash
python PI_HIST_AW_GSIs.py --L 6 --n 10000 --np 10000
```

#### **Ablation Study and Parameter Analysis**
To conduct ablation study and parameter analysis for **PI-HIST (R)** on **PPI**:
```bash
python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act no --norm no
python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act no --norm l2

python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act relu --norm no
python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act relu --norm l2
python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act relu --norm z

python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act tanh --norm no
python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act tanh --norm l2
python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act tanh --norm z

python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act sig --norm no
python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act sig --norm l2
python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act sig --norm z

python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act exp --norm no
python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act exp --norm l2
python PI_HIST_R_NPGp.py --data_name ppi --d 256 --act exp --norm z
```

To conduct ablation study and parameter analysis for **PI-HIST (R)** on **USA**:
```bash
python PI_HIST_R_NIGp.py --data_name usa --d 64 --act no --norm no
python PI_HIST_R_NIGp.py --data_name usa --d 64 --act no --norm l2

python PI_HIST_R_NIGp.py --data_name usa --d 64 --act relu --norm no
python PI_HIST_R_NIGp.py --data_name usa --d 64 --act relu --norm l2
python PI_HIST_R_NIGp.py --data_name usa --d 64 --act relu --norm z

python PI_HIST_R_NIGp.py --data_name usa --d 64 --act tanh --norm no
python PI_HIST_R_NIGp.py --data_name usa --d 64 --act tanh --norm l2
python PI_HIST_R_NIGp.py --data_name usa --d 64 --act tanh --norm z

python PI_HIST_R_NIGp.py --data_name usa --d 64 --act sig --norm no
python PI_HIST_R_NIGp.py --data_name usa --d 64 --act sig --norm l2
python PI_HIST_R_NIGp.py --data_name usa --d 64 --act sig --norm z

python PI_HIST_R_NIGp.py --data_name usa --d 64 --act exp --norm no
python PI_HIST_R_NIGp.py --data_name usa --d 64 --act exp --norm l2
python PI_HIST_R_NIGp.py --data_name usa --d 64 --act exp --norm z
```

To conduct ablation study and parameter analysis for **PI-HIST (A)** on **PPI**:
```bash
python PI_HIST_A_NIGp.py --data_name usa --d 64 --act no --norm no
python PI_HIST_A_NIGp.py --data_name usa --d 64 --act no --norm l2

python PI_HIST_A_NIGp.py --data_name usa --d 64 --act relu --norm no
python PI_HIST_A_NIGp.py --data_name usa --d 64 --act relu --norm l2
python PI_HIST_A_NIGp.py --data_name usa --d 64 --act relu --norm z

python PI_HIST_A_NIGp.py --data_name usa --d 64 --act tanh --norm no
python PI_HIST_A_NIGp.py --data_name usa --d 64 --act tanh --norm l2
python PI_HIST_A_NIGp.py --data_name usa --d 64 --act tanh --norm z

python PI_HIST_A_NIGp.py --data_name usa --d 64 --act sig --norm no
python PI_HIST_A_NIGp.py --data_name usa --d 64 --act sig --norm l2
python PI_HIST_A_NIGp.py --data_name usa --d 64 --act sig --norm z

python PI_HIST_A_NIGp.py --data_name usa --d 64 --act exp --norm no
python PI_HIST_A_NIGp.py --data_name usa --d 64 --act exp --norm l2
python PI_HIST_A_NIGp.py --data_name usa --d 64 --act exp --norm z
```

To conduct ablation study and parameter analysis for **PI-HIST (A)** on **USA**:
```bash
python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act no --norm no
python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act no --norm l2

python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act relu --norm no
python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act relu --norm l2
python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act relu --norm z

python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act tanh --norm no
python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act tanh --norm l2
python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act tanh --norm z

python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act sig --norm no
python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act sig --norm l2
python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act sig --norm z

python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act exp --norm no
python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act exp --norm l2
python PI_HIST_A_NPGp.py --data_name ppi --d 256 --act exp --norm z
```
