# Topology-induced Operators Reveal Complementary Graph Representations without Training

### Abstract
Graph representation learning has largely focused on designing increasingly sophisticated models to transform graph topology into vector representations, or embeddings. However, the extent to which embedding quality depends on model learning, rather than on the underlying topological transformations, remains unclear. Here, we show that informative embeddings can be derived without complicated model design and gradient-based training. Propagating random features through implicit hierarchical structures induced by random walks and anonymous walks yields embeddings that capture node proximity and structural role, respectively. These two training-free embeddings preserve complementary aspects of graph organization and perform competitively with classic and recent methods across various node-, edge-, and graph-level tasks. They often require substantially less computation, resulting in a favorable quality–efficiency trade-off. Combining the two types of embeddings further improves inference quality of some tasks compared with using either embedding type alone. Our results suggest that informative graph embeddings can arise from carefully chosen topological transformations before any learned operation is applied.

### Citing
TBD

If you have any questions regarding this repository, you can contact the author via [mengqin_az@foxmail.com].

### Requirements
TBD

### Usage
***
### Visualizing Main Results
The pre-computed results for the visualization have been placed in ```./res/```. The pre-computed embeddings and features for the visualization of graph-level case studies have been placed in ```./emb/```. Run the following commands to visualize results of experimental evaluations and case studies.

#### Proof-of-concept (Fig. 1)
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

#### Node-level Evaluations (Fig. 2 and Supplementary Fig. S1)
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

#### Edge-level Evaluations (Fig. 3 ans Supplementary Fig. S2)
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

#### Graph-level Case studies (Fig. 4, Supplementary Fig. S3, and Supplementary Fig. S4)
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

#### Ablation Study & Parameter Analysis (Supplementary Fig. S5， S6， S7, and S8)
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

Download and extract the the [pre-processed real-world datasets](https://drive.google.com/file/d/1upLsGSdGFL9eDTdgC5iPYJKBgJ7XrOFu/view?usp=sharing) (~117MB) for edge-level tasks with split training and test sets. Place the extracted files in ```./data_LP/```.

Download the pre-computed AW-induced hierarchical structures for PI-HIST (A) on node- and graph-level tasks. Place the extracted files in ```./AW_hier/```

Download the pre-computed AW-induced hierarchical structures for PI-HIST (A) on edge-level tasks. Place the extracted files in ```./AW_hier_LP/```.

Generate AW-induced hierarchical structures from scratch

***
### Reproducing Main Results

#### Node-level Evaluations

#### Edge-level Evaluations

#### Graph-level Case studies

#### Ablation Study and Parameter Analysis
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
