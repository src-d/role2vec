import os
import tempfile
import unittest

from sklearn.externals import joblib

from role2vec.roles.base import RolesBase


class RolesBaseTests(unittest.TestCase):
    def setUp(self):
        self.model = 1334
        with tempfile.NamedTemporaryFile(delete=False) as model_path:
            self.model_path = model_path.name
            joblib.dump(self.model, self.model_path)
        with tempfile.NamedTemporaryFile() as emb_path:
            self.rb = RolesBase(log_level="INFO", num_processes=1, emb_path=emb_path.name)

    def tearDown(self):
        os.remove(self.model_path)

    def test_save(self):
        with self.assertRaises(ValueError):
            self.rb.save("")
        try:
            self.rb.model = self.model
            with tempfile.NamedTemporaryFile() as model_path:
                self.assertIsNone(self.rb.save(model_path.name))
        finally:
            self.rb.model = None

    def test_load(self):
        with self.assertRaises(ValueError):
            self.rb.load("")
        try:
            self.rb.load(self.model_path)
            self.assertEqual(self.rb.model, self.model)
        finally:
            self.rb.model = None

    def test_train(self):
        with self.assertRaises(NotImplementedError):
            self.rb.train("")

    def test_test(self):
        with self.assertRaises(NotImplementedError):
            self.rb.test("")


if __name__ == "__main__":
    unittest.main()
