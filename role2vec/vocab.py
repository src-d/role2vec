from collections import Counter
from typing import Dict, List

from ast2vec.token_parser import TokenParser
from ast2vec.uast import UASTModel
from role2vec.map_reduce import MapReduce
from role2vec.utils import node_iterator, read_paths, save_vocab


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

    def create(self, files: List[str]) -> Dict[str, int]:
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
                for node, _ in node_iterator(uast):
                    tokens.update(self._get_tokens(node))
            return tokens

        @MapReduce.wrap_queue_out()
        def combine_vocab(self, result):
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


def vocab_entry(args):
    uasts = read_paths(args.input)
    vocab = Vocab(args.log_level, args.processes)
    words = vocab.create(uasts)
    save_vocab(args.output, words)
