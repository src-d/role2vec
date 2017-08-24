from collections import namedtuple
import random

import numpy as np

from ast2vec.pickleable_logger import PickleableLogger
from ast2vec.token_parser import TokenParser

GraphNode = namedtuple("GraphNode", ["id", "neighbors", "tokens"])


class Graph(PickleableLogger):
    def __init__(self, log_level, num_walks, walk_length, p, q):
        assert walk_length > 1, "Random walks have at least two nodes."

        super(Graph, self).__init__(log_level=log_level)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = 1 / p
        self.q = 1 / q
        self.token_parser = TokenParser()

    def node2vec_walk(self, start_node, edges, nodes):
        """
        Simulate a random walk starting from start node.
        """
        walk = [None] * self.walk_length
        walk[0] = start_node
        walk[1] = nodes[start_node.neighbors[int(np.random.rand() * len(start_node.neighbors))]]

        for i in range(2, self.walk_length):
            cur_node = walk[i - 1]
            prev_node = walk[i - 2]
            walk[i] = nodes[cur_node.neighbors[alias_draw(*edges[(prev_node.id, cur_node.id)])]]

        return walk

    def simulate_walks(self, uasts):
        """
        Repeatedly simulate random walks from each node.
        """
        all_walks = []

        for uast, filename in zip(uasts.uasts, uasts.filenames):
            nodes, edges = self._preprocess_uast(uast)
            n_nodes = len(nodes)

            if n_nodes == 1:
                self._log.info(
                    "Skipping UAST for %s: has a single node." % filename)
                continue

            self._preprocess_transition_probs(nodes, edges)
            walks = [None] * (n_nodes * self.num_walks)
            self._log.info("Walk iteration:")

            for walk_iter in range(self.num_walks):
                self._log.info("%d/%d" % (walk_iter + 1, self.num_walks))
                for i, node in enumerate(nodes):
                    walks[n_nodes * walk_iter + i] = self.node2vec_walk(node, edges, nodes)

            all_walks.extend(walks)

        random.shuffle(all_walks)
        return all_walks

    def _get_alias_edge(self, src_id, dst_id, edges, nodes):
        """
        Get the alias edge setup lists for a given edge.
        """
        unnormalized_probs = [
            self.p if dst_nbr == src_id else
            1 if (dst_nbr, src_id) in edges else
            self.q for dst_nbr in nodes[dst_id].neighbors
        ]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [u_prob / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def _get_log_name(self):
        return "Graph"

    def _get_tokens(self, uast_node):
        return ["RoleId_%d" % role for role in uast_node.roles] + \
            list(self.token_parser.process_token(uast_node.token))

    def _preprocess_transition_probs(self, nodes, edges):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        self._log.info("Preprocessing transition probabilities.")
        for edge in edges:
            edges[edge] = self._get_alias_edge(edge[0], edge[1], edges, nodes)

    def _preprocess_uast(self, root):
        """
        Add neighbors information to UAST nodes.
        """
        def create_node(node, id):
            return GraphNode(id=id, neighbors=[], tokens=self._get_tokens(node))

        self._log.info("Preprocessing UAST nodes.")
        root_node = create_node(root, 0)
        edges = {}
        queue = [(root, 0)]
        nodes = [root_node]
        n_nodes = 1

        while queue:
            node, node_idx = queue.pop()
            for child in node.children:
                nodes.append(create_node(child, n_nodes))
                nodes[n_nodes].neighbors.append(node_idx)
                nodes[node_idx].neighbors.append(n_nodes)
                edges[(node_idx, n_nodes)] = edges[(n_nodes, node_idx)] = None
                queue.append((child, n_nodes))
                n_nodes += 1

        return nodes, edges


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    # Sort the data into the outcomes with probabilities that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop through and create little binary mixtures that appropriately allocate the larger
    # outcomes over the overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    # Draw from the overall uniform mixture.
    kk = int(np.random.rand() * len(J))

    # Draw from the binary mixture, either keeping the small one, or choosing the associated
    # larger one.
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
