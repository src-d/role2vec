import argparse
from collections import Counter, deque
from functools import partial
from itertools import chain, islice, product, tee
import logging
import os
from pathlib import Path
import time

from gensim.models import Word2Vec
from modelforge.progress_bar import progress_bar

from ast2vec.uast import UASTModel
from build_vocab import Vocab
from map_reduce import MapReduce
from random_walk import Graph


class Node2Vec(MapReduce):
    MAX_VOCAB_WORDS = 1000000

    def __init__(self, log_level, dimensions, num_processes, vocab_path, window, graph):
        super(Node2Vec, self).__init__(log_level=log_level, num_processes=num_processes)
        self.graph = graph
        self.word2vec = Word2Vec(size=dimensions, window=window, workers=8)
        self.word2vec.build_vocab(Vocab.read_vocab(vocab_path)[:self.MAX_VOCAB_WORDS])

    def train(self, fname, output):
        # print("\n\n----- KEK -----\n\n")
        self._log.info("Scanning %s", fname)
        files = [line.strip() for line in open(fname).readlines()]
        self._log.info("Found %d files", len(files))
        if not files:
            return 0

        self._log.info("Train model.")
        self._train(files)
        self._log.info("Finished training.")

        self._log.info("Saving model.")
        self.word2vec.wv.save_word2vec_format(output)

    def _train(self, files):
        @MapReduce.wrap_queue_in
        def process_uast(self, filename):
            uast = UASTModel().load(filename)
            # print("\n\n----- LOL -----\n\n", filename)
            return self.graph.simulate_walks(uast)

        def train_walks(self, n_tasks, queue_out):
            failures = 0

            def consume(iterator, n):
                """Advance the iterator n-steps ahead. If n is none, consume entirely."""
                # Use functions that consume iterators at C speed.
                if n is None:
                    # feed the entire iterator into a zero-length deque
                    deque(iterator, maxlen=0)
                else:
                    # advance to the empty slice starting at position n
                    next(islice(iterator, n, n), None)

            def window(iterable, n=2):
                """s -> (s0, ...,s(n-1)), (s1, ...,sn), (s2, ..., s(n+1)), ..."""
                iters = tee(iterable, n)
                # Could use enumerate(islice(iters, 1, None), 1) to avoid consume(it, 0), but
                # that's slower for larger window sizes, while saving only small fixed "noop" cost
                for i, it in enumerate(iters):
                    consume(it, i)
                return zip(*iters)

            def batch_stream():
                nonlocal failures
                i = 0
                start = time.time()

                for _ in progress_bar(range(n_tasks), self._log, expected_size=n_tasks):
                    result = queue_out.get()
                    if result:
                        for walk in result:
                            walk = [list(map(str, node.tokens)) for node in walk]
                            for walk_window in window(walk, n=self.word2vec.window):
                                yield list(product(*walk_window))
                                i += 1
                                if i % 10000 == 0:
                                    print(i, time.time() - start)
                    else:
                        failures += 1

            self.word2vec.train(
                batch_stream(),
                total_examples=1000000,
                epochs=self.word2vec.iter)
            return failures

        # walks = []

        # @MapReduce.wrap_queue_out
        # def train_walks(res_walks):
        #     nonlocal walks
        #     res_walks = list(chain.from_iterable(
        #         product(*(map(str, node.tokens) for node in walk)) for walk in res_walks))
        #     walks.extend(res_walks)

        self.parallelize(files, process_uast, train_walks)
        # self.word2vec.train(walks, total_examples=len(walks), epochs=self.word2vec.iter)

    def _get_log_name(self):
        return "Node2Vec"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")
    parser.add_argument("input", help="Input file with UASTs.")
    parser.add_argument("output", help="Path to store the result model.")
    parser.add_argument("--dimensions", default=300, help="Dimensionality of embeddings.")
    parser.add_argument("--processes", type=int, default=1, help="Number of processes.")
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
    node2vec = Node2Vec(args.log_level, args.dimensions, args.processes,
                        args.vocabulary, args.window, graph)
    node2vec.train(args.input, args.output)
