from collections import Counter, deque
from itertools import islice, tee
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np


def node_iterator(root):
    """
    Enumerate UAST nodes using depth-first approach.
    """
    queue = [(root, 0)]
    n_nodes = 1
    while queue:
        node, node_idx = queue.pop()
        yield node, node_idx
        for child in node.children:
            queue.append((child, n_nodes))
            n_nodes += 1


def consume(iterator: Iterator, n: int) -> None:
    """
    Advance the iterator n-steps ahead. If n is none, consume entirely.

    :param iterator: Input iterator.
    :param n: Number of steps.
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def window(iterable: Iterable, n: int=2) -> Iterator:
    """
    Create consecutive windows of elements from iterable.

    :param iterable: Input iterable.
    :param n: Window size.
    :return: Iterator for windows from the input iterable.
    """
    iters = tee(iterable, n)
    for i, it in enumerate(iters):
        consume(it, i)
    return zip(*iters)


def read_embeddings(emb_path: str) -> Tuple[Dict[str, np.array], List[str]]:
    emb = {}
    roles = []

    with open(emb_path) as fin:
        for line in fin:
            word, *vec = line.split("\t")
            emb[word] = np.array(vec, dtype=np.float)
            if word.startswith("RoleId_"):
                roles.append(word)

    roles = {role: i for i, role in enumerate(roles)}
    return emb, roles


def read_paths(fname: str) -> List[str]:
    paths = [line.strip() for line in open(fname).readlines()]
    if not paths:
        raise ValueError("Make sure the file is not empty!")
    return paths


def read_vocab(vocab_path: str, num_words: int=None) -> List[str]:
    with open(vocab_path) as fin:
        words = [line.split(" ")[0] for line in islice(fin, num_words)]
    return words


def save_vocab(vocab_path: str, vocab: Counter[str, int]) -> None:
    with open(vocab_path, "w") as fout:
        fout.write("\n".join(
            map(lambda x: "%s %d".join(x), vocab.most_common())))
