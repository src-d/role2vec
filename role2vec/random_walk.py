from collections import namedtuple
import random
from typing import Dict, Iterator, List, Tuple

import numpy as np

from ast2vec.pickleable_logger import PickleableLogger
from ast2vec.token_parser import TokenParser
from role2vec.utils import node_iterator

GraphNode = namedtuple("GraphNode", ["id", "neighbors", "tokens"])


class Graph(PickleableLogger):
    """
    Generates random walks from UASTs.
    """

    def __init__(self, log_level: str, num_walks: int, walk_length: int, p: float, q: float):
        """
        :param log_level: Log level of Node2Vec.
        :param num_walks: Number of random walks from each node.
        :param walk_length: Random walk length.
        :param p: Controls the likelihood of immediately revisiting previous node.
        :param q: Controls the likelihood of exploring outward nodes.
        """
        if walk_length <= 1:
            raise ValueError("Random walks have at least two nodes.")

        super(Graph, self).__init__(log_level=log_level)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = 1 / p
        self.q = 1 / q
        self.token_parser = TokenParser()

    def node2vec_walk(self, start_node: GraphNode, edges: Dict[Tuple[int, int], None],
                      nodes: List[GraphNode]) -> List[GraphNode]:
        """
        Simulate a random walk starting from start node.

        :param start_node: Starting node for random walk.
        :param edges: Dict for storing mapping from node id pairs to transition probabilities.
        :param nodes: List of UAST nodes.
        :return: List of GraphNodes in random walk.
        """
        walk = [None] * self.walk_length
        prev_node = walk[0] = start_node
        cur_node = walk[1] = nodes[random.choice(start_node.neighbors)]

        for i in range(2, self.walk_length):
            J, q = edges[(prev_node.id, cur_node.id)]
            kk = np.random.randint(len(J))

            # Draw a sample from discrete distribution at constant time.
            if np.random.rand() < q[kk]:
                ind = kk
            else:
                ind = J[kk]

            prev_node = cur_node
            cur_node = walk[i] = nodes[cur_node.neighbors[ind]]

        return walk

    def simulate_walks(self, uasts) -> Iterator[List[GraphNode]]:
        """
        Repeatedly simulate random walks from each node.

        :param uasts: List of UASTs.
        :return: Iterator over random walks generated for the input UASTs.
        """
        for uast, filename in zip(uasts.uasts, uasts.filenames):
            nodes, edges = self._preprocess_uast(uast)
            n_nodes = len(nodes)

            if n_nodes == 1:
                self._log.info("Skipping UAST for %s: has a single node." % filename)
                continue

            self._preprocess_transition_probs(nodes, edges)
            self._log.info("Walk iteration:")

            for walk_iter in range(self.num_walks):
                self._log.info("%d/%d" % (walk_iter + 1, self.num_walks))
                iter_nodes = set(node.id for node in nodes)

                while iter_nodes:
                    node = nodes[random.sample(iter_nodes, 1)[0]]
                    walk = self.node2vec_walk(node, edges, nodes)
                    yield walk

                    for walk_node in walk:
                        if walk_node.id in iter_nodes:
                            iter_nodes.remove(walk_node.id)

    def _get_log_name(self):
        return "Graph"

    def _get_tokens(self, uast_node) -> List[str]:
        """
        Return node tokens.

        :param uast_node: UAST node.
        :return: List of tokens.
        """
        return ["RoleId_%d" % role for role in uast_node.roles] + \
            list(self.token_parser.process_token(uast_node.token))

    def _preprocess_transition_probs(self, nodes: List[GraphNode],
                                     edges: Dict[Tuple[int, int], None]) -> None:
        """
        Preprocessing of transition probabilities for guiding the random walks.

        :param nodes: List of GraphNodes in UAST.
        :param edges: Dict for storing mapping from node id pairs to transition probabilities.
        """
        self._log.info("Preprocessing transition probabilities.")
        for edge in edges:
            unnormalized_probs = np.array([
                self.p if dst_nbr == edge[0] else
                1 if (dst_nbr, edge[0]) in edges else
                self.q for dst_nbr in nodes[edge[1]].neighbors
            ])
            edges[edge] = alias_setup(unnormalized_probs / unnormalized_probs.sum())

    def _preprocess_uast(self, root) -> Tuple[List[GraphNode], Dict[Tuple[int, int], None]]:
        """
        Add neighbors information to UAST nodes.

        :param root: Root node in UAST.
        :return: Nodes and edges in the UAST.
        """
        def create_node(node, id):
            return GraphNode(id=id, neighbors=[], tokens=self._get_tokens(node))

        self._log.info("Preprocessing UAST nodes.")
        edges = {}
        nodes = [create_node(root, 0)]
        n_nodes = 1

        for node, node_idx in node_iterator(root):
            for child in node.children:
                nodes.append(create_node(child, n_nodes))
                nodes[n_nodes].neighbors.append(node_idx)
                nodes[node_idx].neighbors.append(n_nodes)
                edges[(node_idx, n_nodes)] = edges[(n_nodes, node_idx)] = None
                n_nodes += 1

        return nodes, edges


def alias_setup(probs: np.array) -> Tuple[np.array, np.array]:
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with
    -many-discrete-outcomes/
    for details

    :param probs: Discrete distribution.
    :return: Two helper tables.
    """
    K = len(probs)
    q = probs * K
    J = np.zeros(K, dtype=np.int)

    # Sort the data into the outcomes with probabilities that are larger and smaller than 1/K.
    smaller = np.where(q < 1.0)[0]
    larger = np.where(q >= 1.0)[0]
    s_idx = len(smaller) - 1
    l_idx = len(larger) - 1

    # Loop through and create little binary mixtures that appropriately allocate the larger
    # outcomes over the overall uniform mixture.
    while s_idx >= 0 and l_idx >= 0:
        small = smaller[s_idx]
        large = larger[l_idx]
        J[small] = large
        q[large] += q[small] - 1.0

        if q[large] < 1.0:
            smaller[s_idx] = large
            l_idx -= 1
        else:
            s_idx -= 1

    return J, q
