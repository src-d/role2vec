import argparse
from collections import Counter
import logging

from ast2vec.token_parser import TokenParser
from ast2vec.uast import UASTModel
from map_reduce import MapReduce


class Vocab(MapReduce):
    def __init__(self, log_level, num_processes, vocab_path):
        super(Vocab, self).__init__(log_level=log_level, num_processes=num_processes)
        self.token_parser = TokenParser()
        if vocab_path is None:
            self.vocab_path = "vocab.txt"
        else:
            self.vocab_path = vocab_path

    def create(self, files):
        vocab = Counter()

        @MapReduce.wrap_queue_in
        def uasts_vocab(self, filename):
            uast_model = UASTModel().load(filename)
            tokens = Counter()
            for uast in uast_model.uasts:
                nodes = [uast]
                while nodes:
                    node = nodes.pop()
                    tokens.update(self._get_tokens(node))
                    nodes.extend(node.children)
            return tokens

        @MapReduce.wrap_queue_out
        def combine_vocab(result):
            nonlocal vocab
            vocab.update(result)

        self.parallelize(files, uasts_vocab, combine_vocab)
        self.save_vocab(self.vocab_path, vocab)
        return vocab

    def _get_log_name(self):
        return "Vocab"

    def _get_tokens(self, uast_node):
        return ["RoleId_%d" % role for role in uast_node.roles] + \
            list(self.token_parser.process_token(uast_node.token))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")
    parser.add_argument("input", help="Input file with UASTs.")
    parser.add_argument("output", help="Path to store vocabulary.")
    parser.add_argument("--processes", type=int, default=2, help="Number of processes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    uasts = open(args.input).read().split("\n")
    vocab = Vocab(args.log_level, args.processes, args.output)
    vocab.create(uasts)
