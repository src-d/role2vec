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
* The goal is to optimize the sum of probabilities of nodes' neighborhoods w.r.t. feature representation function.
* The authors propose a controllable strategy for efficiently sampling neighborhoods (2nd order Random Walk).
* The learnt node representations could be used in a one-vs-rest classifier for multi-lable classification.

Thoughts: Looks promising, however it might be in favor of dense graph structures.


### Poincare Embeddings for Learning Hierarchical Representations

https://arxiv.org/abs/1705.08039

Details:

* Hyperbolic space is inherently good for modeling hierarchical structure of the data: distance ~ similarity, vector norm ~ place in hierarchy.
* The authors propose to use Poincare 2d ball for modelling hyperbolic space.
* There're some stochastic gradient optimization methods suited for working in this space.

Thoughts: This paper brings up a good idea of using different spaces more suited for hierarchical data. The results are impressive and could be useful to us


### Neural Embeddings of Graphs in Hyperbolic Space

https://arxiv.org/abs/1705.10359v1

Details:

* Basically, this paper is identical to "Poincare Embeddings for Learning Hierarchical Representations"


### Structural Deep Network Embeddings

http://dl.acm.org/citation.cfm?id=2939753

Details:

* Offers an explicit objective function for preserving first-order (pairwise) and second-order (neiborhood-wise) proximity.
* We use a deep autoencoder for embedding adjacency vectors correponding to nodes.
* First-order proximity loss is a weighted sum of squared distances in latent space with coefficients from adjacency matrix – it penalizes distant representations of connected nodes.
* Second-order proximity loss is a variant of l2 reconstruction loss for autoencoder, which penalizes more for non-zero values – this helps enforcing global structure learning. 

Thoughts: Since the node embeddings are derived from adjacency vectors, it's not obvious how to add new vertices to the model, or reuse it for a different graph. However, it's an interesting idea on how to capture nonlinearities in nodes relationships.


### Deep Neural Networks for Learning Graph Representations

http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12423/11715

Details:

* The paper argues that sampling linear sequences of nodes from graphs has two major drawbacks: there's little information about the nodes at the boundaries, and it's not straightforward to determine hyperparameters of such sampling. They suggest to use Random Surfing for derving node representations instead (motivated by PageRank): each node is a weighted sum of probability vectors, which indicate the probability of reaching nodes from the starting node after a certain number of steps, with monotonically decreasing weights (this is a generalization over skip-gram and GloVe).
* The node representations are then used to compute Positive Pointwise Mutual Information matrix (it's been proven to be useful in case of linear dimensionality reduction for word vectors).
* Finally, the rows of PPMI matrix are used to train Stacked Denoising Autoencoder – the latent representations are assigned to nodes.

Thoughts: There aren't many experiments and the model is compared to relatively weak baselines, so I assume this approach isn't a breakthrough. However, it has a nice idea of using PPMI matrix and Denoising AE, which could be interesting to combine with the model in "Structural Deep Network Embeddings".


### Learning Convolutional Neural Networks for Graphs

http://proceedings.mlr.press/v48/niepert16.pdf

Details:

* The authors propose a procedure for (i) determining the node sequences for which neighborhood graphs are created and (ii) computing a normalization of neighborhood graphs.
* Node sequence selection: sort nodes according to some labeling (e.g. color refinement a.k.a. naive vertex classification), then traverse this sequence with some stride and generate receptive fields for each selected node.
    * For each selected node we assemble its neighborhood by BFS.
    * Each neighborhood is normalized to produce a receptieve field: pick neighboring nodes according to the receptive field size and canonize the subgraph for these nodes.
* We can interpret node and edge features as channels, thus we can feed the generated receptive fields to a CNN.

Thoughts: the results don't look that impressive. The paper is mostly focused on generating sequences of nodes, and not much info about neural networks architecture. Don't think it'll be helpful to us.


### Line: Large Scale Information Network Embedding

http://dl.acm.org/citation.cfm?id=2741093

Details:

* Same as in "Structural Deep Network Embeddings" we define an explicit objective function for preserving first-order (pairwise) and second-order (neiborhood-wise) proximity.
* First-order proximity loss is a KL-divergence between two probability distributions: edge distribution and pairwise node distribution (probabilities are defined as sigmoid of dot product of corresponding node vectors).
* Second-order proximity loss is a weighted sum of KL-divergences: for each node we calculate KL-divergence between empirical distribution of nodes (normalized adjacency vector) and conditional node distribution (softmax over sigmoids of dot products of corresponding node vectors). The coefficient signify the *prestige* of a node.
* Optimization is done by Asynchronous Stochastic Gradient Algorithm with edge sampling (depends on edge weights).
* The authors propose to add 'edge weights' between vertices with distance 2 in case of a big number of low degree vertices. This should help to capture second-order proximity. 
* If there's a new vertex and we know its neighbors in our graph, then we could easily compute its representation by minimizing a certain objective function.

Thoughts: The idea of adding 'edge weights' is interesting, but I think Random Walk is better suited for optimizing second-order proximity (in this case it would be really close to node2vec with a different loss function, and since Line can outperform node2vec on some tasks, we could get something interesting out of it). 



### Asymmetric Transitivity Preserving Graph Embedding

https://www.cs.sfu.ca/~jpei/publications/Graph%20Embedding%20KDD16.pdf

Details:

* This paper is dedicated to directed graph embeddings.
* The objective is to find a low rank approximation of proximity matrix (Katz index, Adamic-Adar, etc.), i.e. truncated SVD. 
* Proximity matrix could be represented as a product of two matrices, which allows to use JDGSGD – an effective algorithm for matrix decomposition.

Thoughts: The proposed algorithm (called HOPE) works suprisingly well in graph reconstruction and link prediction problems. Apparently, proximity matrices are good at preserving high order proximity information. We could try to use some non-linear dimensionality reductions to further improve its performance.


### GraRep: Learning Graph Representations with Global Structural Information

http://dl.acm.org/citation.cfm?id=2806512

Details: 

* The authors propose the same loss as in skip-gram, but with Noise Contrastive Estimation.
* Turns out optimizing this loss is equivalent to factorizing PMI for transition probability matrix, thus we could use lower dimensional representation of our nodes.
* We can generate multiple k-step transition probability matrices (it contains probabilities for reaching other vertices in exactly k steps), and concatenate their respective lower dimensional approximations.

Thoughts: Matrix factorization based methods can't learn complex non-linear interactions, unless it's explicitly encoded in the matrix itself. This method overcomes some of these limitations by utilizing info from many transition probability matrices, but it feels that "Deep Neural Networks for Learning Graph Representations" offers a better way to handle non-linear dependencies in data.


### Gated Graph Sequence Neural Networks

https://arxiv.org/abs/1511.05493

Details: 

* Assume we have some node annotations (this is additional info, the annotations could be set to all 0's), then we'll initialize node representations with these annotations padded by zeros (so that our representation space is larger than annotation space) multiplied by the adjacency matrix.
* We feed these representations into a recurrent network with GRU-like units for a fixed number of steps to generate new node representations, and then train an output model on top of these representations (using backpropagation through time). This will be called Gated Graph Neural Networks.
* We could stack GCNNs for generating sequence of node representations. On each of these steps there'll be a separate GCNN trained for predicting outputs. There could be tasks for which we have the intermediate representations during training time, in which case we train all GCNNs separately, otherwise we train them jointly. This will be called Gated Graph Sequence Neural Networks.

Thoughts: Very nice results on the bAbI tasks dataset, which confirm that explicitly modelling graph relations greatly simplifies the learning task. However, this model is not good for scalling, and doesn't provide any idea on how to extend it to new graphs. 


### A Generalization of Convolutional Neural Networks to Graph-Structured Data

http://arxiv.org/abs/1704.08165

Details: 

* Even if we don't know the spatial structure of a graph, we could always construct its transition matrix.
* Define Q as the sum of transition matrix powers up to some power k. Matrix Q could be interpreted as the expected number of visits in k steps.
* Each node is encoded with the top p highest values from the corresponding row in matrix Q. These vectors could be then used for applying convolutions, etc.

Thoughts: There're no experiments for node classification, or link prediction on standard graph datasets, so it's hard to say whether this is a good approach for our task or not. The authors state that this work might be the very first one that generalizes convolutions for any graph, i.e. you could learn it once and then apply it to any graph, given its matrix Q. 


### Dynamics Based Features For Graph Classification

http://arxiv.org/abs/1705.10817

Details: 

* Let's define an indicator vector which will represent random walker's position (coordinates represent nodes). M - transition matrix.
* Compute autocovariance matrix at time step t for the indicator vector, i.e. covariance matrix between the indicator vector at positions 1 and t+1 – it could be expressed in terms of M and stationary distribution under M.
* Let's denote H - the matrix containing all attribute vectors for nodes. For example, we could include second left eigenvector of M as an attribute. If an attribute is categorical it's substituted by a one-hot vector. 
* Compute trace of matrix product between transposed H, autocovariance matrix and H. This is also known as generalized assortativity coefficient. Combining these coefficients computed for different attributes and time steps we'll get a vector representation for our initial graph.

Thoughts: This is paper does graph embedding without producing any node embeddings. Could be interesting for finding similar graphs.


### Distributed Representation of Subgraphs

https://arxiv.org/abs/1702.06921v1

Details: 

* 
