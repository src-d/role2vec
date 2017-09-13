import argparse
from collections import defaultdict
from itertools import product
import logging
import os
import time

import numpy
from scipy.sparse import coo_matrix

from ast2vec.coocc import Cooccurrences
from ast2vec.uast import UASTModel
from map_reduce import MapReduce
from random_walk import Graph
from utils import read_paths, read_vocab


class Node2Vec(MapReduce):
    MAX_VOCAB_WORDS = 1000000

    def __init__(self, log_level, num_processes, vocab_path, window, graph):
        super(Node2Vec, self).__init__(log_level=log_level, num_processes=num_processes)
        self.graph = graph
        self.num_processes = num_processes
        self.vocab = {w: i for i, w in enumerate(
            read_vocab(vocab_path)[:Node2Vec.MAX_VOCAB_WORDS])}
        self.window = window

    def process(self, fname, output):
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
                for i, cur_tokens in enumerate(walk[:-1]):
                    for next_tokens in walk[(i + 1):(i + self.window)]:
                        for word1, word2 in product(cur_tokens, next_tokens):
                            dok_matrix[(word1, word2)] += 1
                            dok_matrix[(word2, word1)] += 1

            del uast

            mat = coo_matrix(
                (Node2Vec.MAX_VOCAB_WORDS, Node2Vec.MAX_VOCAB_WORDS), dtype=numpy.float32)
            mat.row = row = numpy.empty(len(dok_matrix), dtype=numpy.int32)
            mat.col = col = numpy.empty(len(dok_matrix), dtype=numpy.int32)
            mat.data = data = numpy.empty(len(dok_matrix), dtype=numpy.float32)
            for i, (coord, val) in enumerate(sorted(dok_matrix.items())):
                row[i], col[i] = coord
                data[i] = val

            del dok_matrix

            coocc = Cooccurrences()
            coocc.construct(tokens=sorted(self.vocab, key=self.vocab.get), matrix=mat)
            coocc.save(output)
            self._log.info("Finished processing %s", filename)
            return filename

        @MapReduce.wrap_queue_out
        def process_output(self, result):
            pass

        self._log.info("Processing files.")
        paths = self._preprocess_paths(paths, output)
        start_time = time.time()
        self.parallelize(paths, process_uast, process_output)
        self._log.info("Finished processing in %.2f.", time.time() - start_time)

    def _get_log_name(self):
        return "Node2Vec"

    def _preprocess_paths(self, paths, output):
        preprocessed_paths = []
        for p in paths:
            name = os.path.basename(p)
            if name.startswith("uast_"):
                name = name[len("uast_"):]
            out_dir = os.path.join(output, name[0])
            os.makedirs(out_dir, exist_ok=True)
            out_fname = os.path.join(out_dir, name)
            preprocessed_paths.append((p, out_fname))
        return preprocessed_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")
    parser.add_argument("input", help="Input file with UASTs.")
    parser.add_argument("output", help="Path to store the resulting matrices.")
    parser.add_argument("--processes", type=int, default=4, help="Number of processes.")
    parser.add_argument("--vocabulary", default="vocab.txt", help="File with vocabulary.")
    parser.add_argument(
        "-n", "--num-walks", type=int, default=1, help="Number of random walks from each node.")
    parser.add_argument(
        "-l", "--walk-length", type=int, default=80, help="Length of each random walk.")
    parser.add_argument(
        "-w", "--window", type=int, default=5, help="Window size for node context.")
    parser.add_argument(
        "-p", type=float, default=1.0,
        help="Controls the likelihood of immediately revisiting previous node.")
    parser.add_argument(
        "-q", type=float, default=1.0, help="Controls the likelihood of exploring outward nodes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    graph = Graph(args.log_level, args.num_walks, args.walk_length, args.p, args.q)
    node2vec = Node2Vec(args.log_level, args.processes,
                        args.vocabulary, args.window, graph)
    node2vec.process(args.input, args.output)
