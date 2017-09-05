import argparse
import logging

from role_model import RoleModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")
    parser.add_argument("--train", help="Input file with UASTs for training.")
    parser.add_argument("--test", help="Input file with UASTs for testing.")
    parser.add_argument("--model", required=True, help="Path to store trained model.")
    parser.add_argument("--processes", type=int, default=2, help="Number of processes.")
    parser.add_argument("--embeddings", help="File with roles and tokens embeddings.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rm = RoleModel(args.log_level, args.processes, args.embeddings)

    if args.train:
        rm.train(args.train)
        rm.save(args.model)
    else:
        rm.load(args.model)

    if args.test:
        rm.test(args.test)
