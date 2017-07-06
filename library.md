# Papers


### Graph Embedding Techniques, Applications, and Performance: A Survey

https://arxiv.org/abs/1705.02801v2

Details:
* This is a survey of different approaches for embeddings graphs/nodes.
* In a section about node classification they state, that there're two approaches for classification: feature based (features based on network statistics) and random walk based (sample sequences of nodes).

Thoughts: Good source of papers and some interesting comparisons of different algorithms.


### Node2Vec

http://snap.stanford.edu/node2vec/

Details:
* This is a graph-variant of word2vec model.
* The goal is to optimize $\max_f \sum_{u\in V} \log{Pr(N_S(u)|f(u))}$, where $N_S(u)$ - the network neighborhood of node u, f - feature representation.
* The authors propose a controllable strategy for efficiently sampling $N_S(u)$ (2nd order Random Walk)
* The learnt node representations could be used in a one-vs-rest classifier for multi-lable classification.

Thoughts: Looks promising, however it favors dense graph structures.


### Poincare Embeddings for Learning Hierarchical Representations

https://arxiv.org/abs/1705.08039

Details:

* Hyperbolic space is inherently good for modeling hierarchical structure of the data: distance $\sim$ similarity, vector norm $\sim$ place in hierarchy.
* The authors propose to use Poincare 2d ball for modelling hyperbolic space.
* There're some stochastic gradient optimization methods suited working in this space.

Thoughts: This paper brings up a good idea of using different spaces more suited for hierarchical data. The results are impressive and could be useful to us


### Neural Embeddings of Graphs in Hyperbolic Space

https://arxiv.org/abs/1705.10359v1

Details:

* Basically, this paper is identical to "Poincare Embeddings for Learning Hierarchical Representations"


### Structural Deep Network Embeddings

http://dl.acm.org/citation.cfm?id=2939753

Details:

* Offers an explicit objective function for preserving first-order (pairwise) and second-order (neiborhood-wise) proximity.
* We use a deep autoencoder for embedding adjacency vectors correponding to nodes: $s_i=\{s_{i,j}\}_{j=1}^n$.
* First-order proximity loss: $L_1=\sum_{i,j=1}^n s_{i,j} ||y_i-y_j||^2$, where $y_i$ - latent representation.
* Second-order proximity loss: $L_2=\sum_{i=1}^n ||(\hat{x}_i-x_i) \odot b_i||^2$, where if $s_{i,j}=0$, $b_i=1$, else $b_i=\beta$.

Thoughts: Since the node embeddings are derived from adjacency vectors, it's not obvious how to add new vertices to the model, or reuse it for a different graph. However, it's an interesting idea on how to capture nonlinearities in nodes relationships.


### Deep Neural Networks for Learning Graph Representations

http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12423/11715

Details:


### Learning Convolutional Neural Networks for Graphs

http://proceedings.mlr.press/v48/niepert16.pdf

Details:


### Line: Large Scale Information Network Embedding

http://dl.acm.org/citation.cfm?id=2741093

Details:


### Asymmetric Transitivity Preserving Graph Embedding

https://www.cs.sfu.ca/~jpei/publications/Graph%20Embedding%20KDD16.pdf

Details:


### GraRep: Learning Graph Representations with Global Structural Information

http://dl.acm.org/citation.cfm?id=2806512

Details


