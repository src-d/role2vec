from itertools import islice
from typing import Dict, List, Tuple

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
    with open(fname) as fin:
        paths = [line.strip() for line in fin.readlines()]
    if not paths:
        raise ValueError("Make sure the file is not empty!")
    return paths


def read_vocab(vocab_path: str, num_words: int=None) -> List[str]:
    with open(vocab_path) as fin:
        words = [line.split(" ")[0] for line in islice(fin, num_words)]
    return words


def save_vocab(vocab_path: str, vocab: Dict[str, int]) -> None:
    with open(vocab_path, "w") as fout:
        fout.write("\n".join(map(lambda x: "%s %d" % x, vocab.most_common())))
