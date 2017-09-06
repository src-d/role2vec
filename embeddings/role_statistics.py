import argparse
from collections import Counter
import json
import logging

from ast2vec.uast import UASTModel
from map_reduce import MapReduce


class RoleStat(MapReduce):
    def calc(self, fname, stat_output, susp_output):
        paths = self.read_paths(fname)
        global_counter = Counter()
        suspicious = []

        @MapReduce.wrap_queue_in
        def process_uast(self, filename):
            counter = Counter()
            uast_model = UASTModel().load(filename)
            for uast in uast_model.uasts:
                queue = [uast]
                counter[len(uast.roles)] += 1
                while queue:
                    node = queue.pop()
                    counter[len(node.roles)] += 1
                    queue.extend(node.children)
            return counter, filename

        @MapReduce.wrap_queue_out
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")
    parser.add_argument("input", help="Input file with UASTs.")
    parser.add_argument("--stat", help="Path to store resulting statisics.")
    parser.add_argument("--susp", help="Path to store suspicious UASTs.")
    parser.add_argument("--processes", type=int, default=4, help="Number of processes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    role_stat = RoleStat(args.log_level, args.processes)
    role_stat.calc(args.input, args.stat, args.susp)
