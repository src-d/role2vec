from collections import namedtuple
from itertools import chain

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

from ast2vec.token_parser import TokenParser
from ast2vec.uast import UASTModel
from map_reduce import MapReduce

Node = namedtuple("Node", ["id", "parent", "children", "roles", "tokens"])


class RoleModel(MapReduce):
    def __init__(self, log_level, num_processes, emb_path, model_path):
        super(RoleModel, self).__init__(log_level=log_level, num_processes=num_processes)
        self.emb, self.roles = self.load_emb(emb_path)
        self.model = None
        self.path = model_path
        self.token_parser = TokenParser()

    def train(self, fname):
        self._log.info("Scanning %s", fname)
        files = [line.strip() for line in open(fname).readlines()]
        self._log.info("Found %d files", len(files))
        if not files:
            return 0

        self._log.info("Train model.")
        self.model = self._train(files)
        self._log.info("Finished training.")

        self._log.info("Saving model.")
        joblib.dump(self.model, self.path)

    def test(self):
        self._log.info("Loading model.")
        self.model = joblib.load(self.path)

    def _train(self, files):
        model = MLPClassifier(random_state=1, verbose=True)
        dummies = [DummyClassifier(s, random_state=1)
                   for s in ["stratified", "most_frequent", "uniform"]]
        model.classes_ = sorted(self.roles.values())
        # classes = sorted(self.roles.values())

        @MapReduce.wrap_queue_in
        def process_uast(self, filename):
            X, y = [], []
            uast_model = UASTModel().load(filename)

            for uast in uast_model.uasts:
                queue = [(uast, 0)]
                node_vecs = [self.mean_vec([uast])]
                n_nodes = 1

                while queue:
                    node, node_idx = queue.pop()
                    for child in node.children:
                        child_vec = self.mean_vec([child])
                        # add child to dataset
                        if child.children and child_vec is not None:
                            labels = np.zeros(len(self.roles), dtype=np.int8)
                            labels[[self.roles["RoleId_%d" % role] for role in child.roles]] = 1
                            X.append(np.concatenate(
                                (self.mean_vec(child.children), node_vecs[node_idx])))
                            y.append(labels)
                            queue.append((child, n_nodes))
                            node_vecs.append(child_vec)
                            n_nodes += 1

            return X, y

        data_X, data_y = [], []
        @MapReduce.wrap_queue_out
        def train_uast(result):
            nonlocal model, data_X, data_y
            X, y = result
            data_X.extend(X), data_y.extend(y)
            # model.partial_fit(X, y, classes)
            # print(model.loss_)

        self.parallelize(files, process_uast, train_uast)
        np.savetxt("X.txt", data_X)
        np.savetxt("y.txt", data_y)
        # model.fit(data_X, data_y)
        # for d in dummies:
        #     d.fit(data_X, data_y)
        # print(model.score(data_X, data_y), *(d.score(data_X, data_y) for d in dummies))
        return model

    def mean_vec(self, nodes):
        vecs = [self.emb[t] for node in nodes for t in chain(node.token,
                ["RoleId_%d" % role for role in node.roles]) if t in self.emb]
        if vecs:
            return np.mean(vecs, axis=0)
        return None

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
