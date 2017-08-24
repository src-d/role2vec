import argparse
from collections import Counter
import logging
import os
from pathlib import Path
import struct

from ast2vec.coocc import Cooccurrences
from build_vocab import Vocab
from map_reduce import MapReduce


class GloVe(MapReduce):
    def __init__(self, log_level, num_processes, vocab_path):
        super(GloVe, self).__init__(log_level=log_level, num_processes=num_processes)
        self.vocab = {word: i for i, word in enumerate(Vocab.read_vocab(vocab_path))}

    def convert(self, src_dir, output, file_filter):
        self._log.info("Scanning %s", src_dir)
        files = [str(p) for p in Path(src_dir).glob(file_filter)]
        self._log.info("Found %d files", len(files))
        if not files:
            return 0

        self._log.info("Combine proximity matrices.")
        mat = self.extract(files)
        self._log.info("Finished combining.")

        self._log.info("Saving matrix.")
        self.save_mat(mat, output)

    def extract(self, files):
        counter = Counter()

        @MapReduce.wrap_queue_in
        def process_prox(self, filename):
            prox = Cooccurrences().load(filename)
            return {(prox.tokens[i], prox.tokens[j]): val for
                    i, j, val in zip(prox.matrix.row, prox.matrix.col, prox.matrix.data)}

        @MapReduce.wrap_queue_out
        def combine_prox(result):
            nonlocal counter
            counter.update(
                {(self.vocab[i], self.vocab[j]): val for (i, j), val in result.items()
                 if i in self.vocab and j in self.vocab})

        self.parallelize(files, process_prox, combine_prox)
        return counter

    @staticmethod
    def save_mat(mat, output):
        with open(output, "wb") as fout:
            for (i, j), val in mat.items():
                fout.write(struct.pack("iid", i, j, int(val)))

    def _get_log_name(self):
        return "GloVe"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")
    parser.add_argument("input", help="Input directory with proximity matrices.")
    parser.add_argument("output", help="Path to store combined proximity matrix.")
    parser.add_argument("--filter", default="**/*.asdf", help="File name glob selector.")
    parser.add_argument("--processes", type=int, default=2, help="Number of processes.")
    parser.add_argument("--vocabulary", default="vocab.txt", help="File with vocabulary.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    glove = GloVe(args.log_level, args.processes, args.vocabulary)
    glove.convert(args.input, args.output, args.filter)
