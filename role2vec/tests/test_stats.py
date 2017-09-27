import json
import tempfile
import unittest

from role2vec.stats import RolesStats
import role2vec.tests.models as paths


class RolesStatsTests(unittest.TestCase):
    def setUp(self):
        self.rs = RolesStats(log_level="INFO", num_processes=1)

    def test_calc(self):
        with tempfile.NamedTemporaryFile() as stat, tempfile.NamedTemporaryFile() as susp:
            self.rs.calc(paths.UAST_FILE, stat.name, susp.name)
            role_stats = json.loads(stat.read().decode("utf8"))
            self.assertEqual(role_stats, {"0": 1, "1": 498, "2": 830, "3": 1634, "4": 1407,
                                          "5": 412, "6": 718, "7": 2, "8": 4, "10": 359,
                                          "11": 411})


if __name__ == "__main__":
    unittest.main()
