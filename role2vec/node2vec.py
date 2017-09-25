from collections import defaultdict
from itertools import product
import os
from typing import List

import numpy
from scipy.sparse import coo_matrix, diags

from ast2vec.coocc import Cooccurrences
from ast2vec.uast import UASTModel
from map_reduce import MapReduce
from random_walk import Graph
from utils import read_paths, read_vocab


class Node2Vec(MapReduce):
    """
    Uses Node2Vec random walk algorithm for assembling proximity matrices from UASTs.
    Refer to https://github.com/aditya-grover/node2vec
    """

    MAX_VOCAB_WORDS = 1000000

    def __init__(self, log_level: str, num_processes: int, vocab_path: str, window: int,
                 graph: Graph):
        """
        :param log_level: Log level of Node2Vec.
        :param num_processes: Number of running processes. There's always one additional process
                              for reducing data.
        :param vocab_path: Path to stored vocabulary.
        :param window: Context window size for collecting proximities.
        :param graph: Graph object for random walks generation.
        """
        super(Node2Vec, self).__init__(log_level=log_level, num_processes=num_processes)
        self.graph = graph
        self.vocab = {w: i for i, w in enumerate(read_vocab(vocab_path, Node2Vec.MAX_VOCAB_WORDS))}
        self.window = window

    def process(self, fname: str, output_dir: str) -> None:
        """
        Extract proximity matrices from UASTs.

        :param fname: Path to file with filepaths to stored UASTs.
        :param output_dir: Path to directory for storing proximity matrices.
        """
        self._log.info("Scanning %s", fname)
        paths = read_paths(fname)
        self._log.info("Found %d files", len(paths))

        @MapReduce.wrap_queue_in
        def process_uast(self, obj):
            filename, output = obj
            self._log.info("Processing %s", filename)
            uast = UASTModel().load(filename)
            dok_matrix = defaultdict(int)

            for walk in self.graph.simulate_walks(uast):
                walk = [[self.vocab[t] for t in map(str, node.tokens)
                        if t in self.vocab] for node in walk]
                # Connect each token to the next `self.window` tokens.
                for i, cur_tokens in enumerate(walk[:-1]):
                    for next_tokens in walk[(i + 1):(i + self.window)]:
                        for word1, word2 in product(cur_tokens, next_tokens):
                            # Symmetry will be accounted for later
                            dok_matrix[(word1, word2)] += 1

            del uast

            mat = coo_matrix(
                (Node2Vec.MAX_VOCAB_WORDS, Node2Vec.MAX_VOCAB_WORDS), dtype=numpy.int32)
            mat.row = row = numpy.empty(len(dok_matrix), dtype=numpy.int32)
            mat.col = col = numpy.empty(len(dok_matrix), dtype=numpy.int32)
            mat.data = data = numpy.empty(len(dok_matrix), dtype=numpy.int32)
            for i, (coord, val) in enumerate(sorted(dok_matrix.items())):
                row[i], col[i] = coord
                data[i] = val

            del dok_matrix
            # Accounting for symmetry
            mat = coo_matrix(mat + mat.T - diags(mat.diagonal()))

            coocc = Cooccurrences()
            coocc.construct(tokens=sorted(self.vocab, key=self.vocab.get), matrix=mat)
            coocc.save(output)
            self._log.info("Finished processing %s", filename)
            return filename

        @MapReduce.wrap_queue_out
        def process_output(self, result):
            pass

        self._log.info("Preprocessing file names.")
        paths = self._preprocess_paths(paths, output_dir)
        self.parallelize(paths, process_uast, process_output)

    def _get_log_name(self):
        return "Node2Vec"

    def _preprocess_paths(self, paths: List[str], output_dir: str) -> List[str]:
        """
        Prepare paths for storing proximity matrices.

        :param paths: List of filepaths to stored UASTs.
        :param output_dir: Path to directory for storing proximity matrices.
        :return: List of filepaths for storing proximity matrices.
        """
        preprocessed_paths = []
        for p in paths:
            name = os.path.basename(p)
            if name.startswith("uast_"):
                name = name[len("uast_"):]
            out_dir = os.path.join(output_dir, name[0])
            os.makedirs(out_dir, exist_ok=True)
            out_fname = os.path.join(out_dir, name)
            preprocessed_paths.append((p, out_fname))
        return preprocessed_paths


def node2vec_entry(args):
    graph = Graph(args.log_level, args.num_walks, args.walk_length, args.p, args.q)
    node2vec = Node2Vec(args.log_level, args.processes, args.vocabulary, args.window, graph)
    node2vec.process(args.input, args.output)
