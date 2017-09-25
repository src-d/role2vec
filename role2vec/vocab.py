import argparse
from collections import Counter
import logging
from typing import List

from ast2vec.token_parser import TokenParser
from ast2vec.uast import UASTModel
from map_reduce import MapReduce
from utils import save_vocab


class Vocab(MapReduce):
    """
    Collects vocabulary from UASTs.
    """

    def __init__(self, log_level: str, num_processes: int):
        """
        :param log_level: Log level of Vocab.
        :param num_processes: Number of running processes. There's always one additional process
                              for reducing data.
        """
        super(Vocab, self).__init__(log_level=log_level, num_processes=num_processes)
        self.token_parser = TokenParser()

    def create(self, files: List[str]) -> Counter[str, int]:
        """
        Create vocabulary by processing supplied UASTs.

        :param files: List of filepaths to stored UASTs.
        :return: Dict with tokens and their number of occurrences.
        """
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
        return vocab

    def _get_log_name(self):
        return "Vocab"

    def _get_tokens(self, uast_node) -> List[str]:
        """
        Return node tokens.

        :param uast_node: UAST node.
        :return: List of tokens.
        """
        return ["RoleId_%d" % role for role in uast_node.roles] + \
            list(self.token_parser.process_token(uast_node.token))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")
    parser.add_argument("input", help="Input file with UASTs.")
    parser.add_argument("output", default="vocab.txt", help="Path to store vocabulary.")
    parser.add_argument("--processes", type=int, default=2, help="Number of processes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    uasts = open(args.input).read().split("\n")
    vocab = Vocab(args.log_level, args.processes, args.output)
    words = vocab.create(uasts)
    save_vocab(args.output, words)
