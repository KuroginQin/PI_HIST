# Disentangling Topology Structures Improves Graph Inference: From Random Noise to Position and Identity Embedding

### Abstract
Modern graph inference models commonly rely on graph embeddings, which represent graph entities as low-dimensional vectors with certain properties preserved. Despite their success in various tasks, how these models capture different aspects of topology remains not well understood. Node positions and identities are two fundamental yet distinct topology properties associated with community structure and nodes' structural role, respectively. Here, we systematically investigate embeddings preserving such two properties from a new perspective of implicit hierarchical structures induced by graph stochastic processes, with random walk (RW) and anonymous walk (AW) as examples. Surprisingly, we find that a single feedforward propagation of random noise through the RW- and AW-induced hierarchical structure, combined with simple normalization and nonlinear activation, can derive informative position and identity embeddings even without training. We further validate this extreme design by comparing it with $18$ classic and state-of-the-art baselines across $7$ node-, edge-, and graph-level inference tasks on $8$ real and $10$ synthetic datasets. In addition to achieving a favorable quality–efficiency trade-off, our method also demonstrates that a simple combination of position and identity embeddings can obtain better inference quality than either embedding type alone for some tasks. Collectively, this study provides new insights into simple model designs for informative embedding derivation and shows that disentangling topology into complementary aspects with respect to its diversity improves graph inference.

### Citing
TBD

If you have any questions regarding this repository, you can contact the author via [mengqin_az@foxmail.com].

### Requirements
TBD

### Usage
TBD