from collections import Counter
from pathlib import Path
import struct
from typing import Dict, List, Tuple

from ast2vec.coocc import Cooccurrences
from role2vec.map_reduce import MapReduce
from role2vec.utils import read_vocab


class GloVe(MapReduce):
    """
    Converts proximity matrices into GloVe suitable format.
    Refer to https://github.com/stanfordnlp/GloVe
    """

    def __init__(self, log_level: str, num_processes: int, vocab_path: str):
        """
        :param log_level: Log level of GloVe.
        :param num_processes: Number of running processes. There's always one additional process
                              for reducing data.
        :param vocab_path: Path to stored vocabulary.
        """
        super(GloVe, self).__init__(log_level=log_level, num_processes=num_processes)
        self.vocab = {word: i for i, word in enumerate(read_vocab(vocab_path))}

    def convert(self, src_dir: str, output: str, file_filter: str) -> None:
        """
        Combine all proximity matrices and save them into GloVe suitable format.

        :param src_dir: Path to stored proximity matrices.
        :param output: Path for storing the resulting GloVe suitable matrix.
        :param file_filter: Pattern for recursively scanning `src_dir`.
        """
        self._log.info("Scanning %s", src_dir)
        files = [str(p) for p in Path(src_dir).glob(file_filter)]
        self._log.info("Found %d files", len(files))
        if not files:
            return 0

        self._log.info("Combine proximity matrices.")
        mat = self.combine_mats(files)
        self._log.info("Finished combining.")

        self._log.info("Saving matrix.")
        self.save_mat(mat, output)

    def combine_mats(self, files: List[str]) -> Dict[Tuple[str, str], int]:
        """
        Combine proximity matrices.

        :param files: List of filepaths to stored proximity matrices.
        :return: Mapping from token pairs to their proximity combined over all matrices.
        """
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
    def save_mat(mat: Dict[Tuple[str, str], int], output: str) -> None:
        """
        Save matrix in GloVe suitable format.

        :param mat: Counter storing proximities.
        :param output: Path for storing the resulting GloVe suitable matrix.
        """
        with open(output, "wb") as fout:
            for (i, j), val in mat.items():
                fout.write(struct.pack("iid", i, j, int(val)))

    def _get_log_name(self):
        return "GloVe"


def glove_entry(args):
    glove = GloVe(args.log_level, args.processes, args.vocabulary)
    glove.convert(args.input, args.output, args.filter)
