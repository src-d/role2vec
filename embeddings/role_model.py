from collections import namedtuple
from itertools import chain
import os
import time

import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

from ast2vec.token_parser import TokenParser
from ast2vec.uast import UASTModel
from map_reduce import MapReduce

Node = namedtuple("Node", ["id", "parent", "children", "roles", "tokens"])


class RoleModel(MapReduce):
    def __init__(self, log_level, num_processes, emb_path):
        super(RoleModel, self).__init__(log_level=log_level, num_processes=num_processes)
        self.emb, self.roles = self.load_emb(emb_path)
        self.model = None
        self.token_parser = TokenParser()

    def save(self, model_path):
        if self.model is None:
            raise ValueError("Model is empty.")
        self._log.info("Saving model to %s.", model_path)
        joblib.dump(self.model, model_path)

    def load(self, model_path):
        if not os.path.exists(model_path):
            raise ValueError("Provided path to model doesn't exist: %s", model_path)
        self.model = joblib.load(model_path)

    def train(self, fname):
        files = self.read_paths(fname)

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

        self.parallelize(files, _process_uast, train_uast)
        self._log.info("Finished training.")

    def test(self, fname):
        files = self.read_paths(fname)

        self._log.info("Test model.")
        y_real, y_pred = [], []

        @MapReduce.wrap_queue_out
        def test_uast(self, result):
            nonlocal y_real, y_pred
            X, y = result
            y_real.extend(y)
            y_pred.extend(self.model.predict_proba(X))

        self.parallelize(files, _process_uast, test_uast)
        np.save("y_real.npy", y_real)
        np.save("y_pred.npy", y_pred)
        self._log.info("Finished testing.")

    @staticmethod
    def load_emb(emb_path):
        emb = {}
        roles = []

        with open(emb_path) as fin:
            for line in fin:
                word, *vec = line.split("\t")
                emb[word] = np.array(vec, dtype=np.float)
                if word.startswith("RoleId_"):
                    roles.append(word)

        roles = {role: i for i, role in enumerate(roles)}
        return emb, roles

    def _mean_vec(self, nodes):
        tokens = [t for node in nodes for t in chain(node.token,
                  ["RoleId_%d" % role for role in node.roles]) if t in self.emb]
        if not tokens:
            return None, 0
        return np.mean([self.emb[t] for t in tokens], axis=0), len(tokens)


@MapReduce.wrap_queue_in
def _process_uast(self, filename):
    X, y = [], []
    uast_model = UASTModel().load(filename)

    for uast in uast_model.uasts:
        queue = [(uast, 0)]
        node_vecs = [self._mean_vec([uast])]
        n_nodes = 1

        while queue:
            node, node_idx = queue.pop()
            for child in node.children:
                child_vec = self._mean_vec([child])
                grandchild_vec = self._mean_vec(child.children)
                # add child to dataset
                if child_vec is not None and grandchild_vec is not None:
                    labels = np.zeros(len(self.roles), dtype=np.int8)
                    labels[[self.roles["RoleId_%d" % role] for role in child.roles]] = 1
                    X.append(np.concatenate((grandchild_vec, node_vecs[node_idx])))
                    y.append(labels)
                    queue.append((child, n_nodes))
                    node_vecs.append(child_vec)
                    n_nodes += 1

    return X, y
