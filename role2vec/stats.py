from collections import Counter
import json

from ast2vec.uast import UASTModel
from role2vec.map_reduce import MapReduce
from role2vec.utils import node_iterator, read_paths


class RolesStats(MapReduce):
    """
    Collects statistics for number of nodes w.r.t. number of node roles in all UASTs.
    """

    def calc(self, fname: str, stat_output: str, susp_output: str) -> None:
        """
        Compute statistics and store them in JSON format.

        :param fname: Path to file with filepaths to stored UASTs.
        :param stat_output: Path for storing JSON file with statistics.
        :param susp_output: Path for storing txt file with info about suspicious UASTs. The file
                            has three columns: filepath to UAST, number of nodes in UAST, number of
                            nodes without roles in UAST.
        """
        paths = read_paths(fname)
        global_counter = Counter()
        suspicious = []

        @MapReduce.wrap_queue_in
        def process_uast(self, filename):
            counter = Counter()
            uast_model = UASTModel().load(filename)
            for uast in uast_model.uasts:
                for node, _ in node_iterator(uast):
                    counter[len(node.roles)] += 1
            return counter, filename

        @MapReduce.wrap_queue_out()
        def combine_stat(self, result):
            nonlocal global_counter
            counter, filename = result
            global_counter.update(counter)
            if 0 in counter:
                suspicious.append((filename, sum(counter.values()), counter[0]))

        self.parallelize(paths, process_uast, combine_stat)
        with open(stat_output, "w") as fout:
            json.dump(global_counter, fout)
        with open(susp_output, "w") as fout:
            for susp_entry in suspicious:
                fout.write(", ".join(map(str, susp_entry)) + "\n")
        self._log.info("Finished collecting statistics.")


def stats_entry(args):
    role_stat = RolesStats(args.log_level, args.processes)
    role_stat.calc(args.input, args.stat, args.susp)
