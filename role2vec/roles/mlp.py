from itertools import chain
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier

from ast2vec.uast import UASTModel
from role2vec.map_reduce import MapReduce
from role2vec.roles_base import register_roles_model, RolesBase
from role2vec.utils import node_iterator, read_paths


@register_roles_model
class RolesMLP(RolesBase):
    """
    Predicts roles using Multi-Layer Perceptron.
    """

    def train(self, fname: str) -> None:
        """
        Train model.

        :param fname: Path to train file with filepaths to stored UASTs.
        """
        paths = read_paths(fname)

        self._log.info("Train model.")
        self.model = MLPClassifier(random_state=1, verbose=True)
        self.model.classes_ = sorted(self.roles.values())
        counter = 0
        start = time.time()

        @MapReduce.wrap_queue_out
        def train_uast(self, result):
            nonlocal counter, start
            X, y = result
            counter += 1
            self.model.partial_fit(X, y)
            print(self.model.loss_, time.time() - start, counter)

        self.parallelize(paths, _process_uast, train_uast)
        self._log.info("Finished training.")

    def test(self, fname: str) -> None:
        """
        Test model.

        :param fname: Path to test file with filepaths to stored UASTs.
        """
        paths = read_paths(fname)

        self._log.info("Test model.")
        y_real, y_pred = [], []

        @MapReduce.wrap_queue_out
        def test_uast(self, result):
            nonlocal y_real, y_pred
            X, y = result
            y_real.extend(y)
            y_pred.extend(self.model.predict_proba(X))

        self.parallelize(paths, _process_uast, test_uast)
        np.save("y_real.npy", y_real)
        np.save("y_pred.npy", y_pred)
        self._log.info("Finished testing.")

    def _mean_vec(self, node) -> Tuple[np.array, int]:
        """
        Calculate mean of role/token embeddings for a node.

        :param node: UAST node.
        :return: Mean of role/token embeddings and their total number.
        """
        tokens = [t for t in chain(node.token, ("RoleId_%d" % role for role in node.roles))
                  if t in self.emb]
        if not tokens:
            return None, 0
        return np.mean([self.emb[t] for t in tokens], axis=0), len(tokens)

    def _mean_vecs(self, root) -> Tuple[Dict[int, np.array], Dict[int, np.array]]:
        """
        Calculate mean of role/token embeddings for nodes and their children in a UAST.

        :param root: UAST root node.
        :return: Mappings from node indices to their parent's and their childrens' mean role/token
                 embeddings.
        """
        node_vecs = {0: self._mean_vec(root)}
        child_vecs = {}
        parent_vecs = {0: None}
        n_nodes = 1  # incremented in accoradance with node_iterator

        for node, node_idx in node_iterator(root):
            node_child_vecs = []
            node_child_ns = []

            for child in node.children:
                child_vec = self._mean_vec(child)
                node_vecs[n_nodes] = child_vec
                parent_vecs[n_nodes] = node_vecs[node_idx][0]
                node_child_vecs.append(child_vec[0])
                node_child_ns.append(child_vec[1])
                n_nodes += 1

            node_child_vecs = list(filter(lambda x: x is not None, node_child_vecs))
            node_child_ns = list(filter(lambda x: x != 0, node_child_ns))

            if node_child_vecs:
                child_vecs[node_idx] = np.average(node_child_vecs, axis=0, weights=node_child_ns)
            else:
                child_vecs[node_idx] = None

        return child_vecs, parent_vecs


@MapReduce.wrap_queue_in
def _process_uast(self, filename: str) -> Tuple[np.array, np.array]:
    """
    Convert UAST into feature and label arrays.
    Had to be defined outside of RolesMLP so that we don't suppply `self` twice.

    :param filename: Path to stored UAST.
    :return: Array of concatenated mean parent and children role/token embeddings for each node and
             the corresponding array of node roles.
    """
    X, y = [], []
    uast_model = UASTModel().load(filename)

    for uast in uast_model.uasts:
        child_vecs, parent_vecs = self._mean_vecs(uast)
        for node, node_idx in node_iterator(uast):
            child_vec = child_vecs[node_idx]
            parent_vec = parent_vecs[node_idx]
            if child_vec is not None and parent_vec is not None:
                labels = np.zeros(len(self.roles), dtype=np.int8)
                labels[[self.roles["RoleId_%d" % role] for role in node.roles]] = 1
                X.append(np.concatenate((child_vec, parent_vec)))
                y.append(labels)

    return np.array(X), np.array(y)
