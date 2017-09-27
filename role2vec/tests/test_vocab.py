import unittest

from role2vec.vocab import Vocab
import role2vec.tests.models as paths


class VocabTests(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocab(log_level="INFO", num_processes=1)
        self.words_true = {}
        with open(paths.VOCAB) as fin:
            for line in fin:
                word, count = line.split()
                self.words_true[word] = int(count)

    def test_create(self):
        words = self.vocab.create([paths.UAST])
        self.assertEqual(len(words), 539)
        self.assertEqual(words, self.words_true)


if __name__ == "__main__":
    unittest.main()
