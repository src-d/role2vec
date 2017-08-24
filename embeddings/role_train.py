import argparse
import logging

from role_model import RoleModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")
    parser.add_argument("input", help="Input file with UASTs.")
    parser.add_argument("output", help="Path to store trained model.")
    parser.add_argument("--processes", type=int, default=2, help="Number of processes.")
    parser.add_argument("--embeddings", help="File with roles and tokens embeddings.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rm = RoleModel(args.log_level, args.processes, args.embeddings, args.output)
    rm.train(args.input)
