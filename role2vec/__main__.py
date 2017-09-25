import argparse
import logging
import sys

from ast2vec.__main__ import ArgumentDefaultsHelpFormatterNoNone, one_arg_parser
from modelforge.logs import setup_logging
from role2vec.glove import glove_entry
from role2vec.node2vec import node2vec_entry
from role2vec.roles.base import roles_entry


def get_parser() -> argparse.ArgumentParser:
    """
    Create main parser.

    :return: Parser
    """
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")

    # Create all common arguments

    process_arg = one_arg_parser("--processes", type=int, default=2, help="Number of processes.")
    vocab_arg = one_arg_parser("--vocabulary", default="vocab.txt", help="File with vocabulary.")

    # Construct subparsers

    subparsers = parser.add_subparsers(help="Commands", dest="command")

    glove_parser = subparsers.add_parser(
        "glove", help="Convert proximity matrices into GloVe suitable format. Refer to "
        "https://github.com/stanfordnlp/GloVe",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[process_arg, vocab_arg])
    glove_parser.set_defaults(handler=glove_entry)
    glove_parser.add_argument("input", help="Input directory with proximity matrices.")
    glove_parser.add_argument("output", help="Path to store combined proximity matrix.")
    glove_parser.add_argument("--filter", default="**/*.asdf", help="File name glob selector.")

    node2vec_parser = subparsers.add_parser(
        "node2vec", help="Node2Vec random walk algorithm for assembling proximity matrices from "
        "UASTs. Refer to https://github.com/aditya-grover/node2vec",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[process_arg, vocab_arg])
    node2vec_parser.set_defaults(handler=node2vec_entry)
    node2vec_parser.add_argument("input", help="Input file with UASTs.")
    node2vec_parser.add_argument("output", help="Path to store the resulting matrices.")
    node2vec_parser.add_argument(
        "-n", "--num-walks", type=int, default=1, help="Number of random walks from each node.")
    node2vec_parser.add_argument(
        "-l", "--walk-length", type=int, default=80, help="Length of each random walk.")
    node2vec_parser.add_argument(
        "-w", "--window", type=int, default=5, help="Window size for node context.")
    node2vec_parser.add_argument(
        "-p", type=float, default=1.0,
        help="Controls the likelihood of immediately revisiting previous node.")
    node2vec_parser.add_argument(
        "-q", type=float, default=1.0, help="Controls the likelihood of exploring outward nodes.")

    roles_parser = subparsers.add_parser(
        "mlp", help="Predict roles using Multi-Layer Perceptron.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[process_arg])
    roles_parser.set_defaults(handler=roles_entry)
    roles_parser.add_argument("algorithm", help="Specify training algorithm.")
    roles_parser.add_argument("--train", help="Input file with UASTs for training.")
    roles_parser.add_argument("--test", help="Input file with UASTs for testing.")
    roles_parser.add_argument("--model", required=True, help="Path to store trained model.")
    roles_parser.add_argument(
        "--embeddings", required=True, help="File with roles and tokens embeddings.")

    return parser


def main():
    """
    Create all the argparsers and invoke the function from set_defaults().

    :return: The result of the function from set_defaults().
    """
    parser = get_parser()
    args = parser.parse_args()
    args.log_level = logging._nameToLevel[args.log_level]
    setup_logging(args.log_level)
    try:
        handler = args.handler
    except AttributeError:
        def print_usage(_):
            parser.print_usage()

        handler = print_usage
    return handler(args)

if __name__ == "__main__":
    sys.exit(main())
