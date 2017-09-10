from collections import deque
from itertools import islice, tee


def consume(iterator, n):
    """Advance the iterator n-steps ahead. If n is none, consume entirely."""
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def window(iterable, n=2):
    """s -> (s0, ...,s(n-1)), (s1, ...,sn), (s2, ..., s(n+1)), ..."""
    iters = tee(iterable, n)
    # Could use enumerate(islice(iters, 1, None), 1) to avoid consume(it, 0), but
    # that's slower for larger window sizes, while saving only small fixed "noop" cost
    for i, it in enumerate(iters):
        consume(it, i)
    return zip(*iters)


def read_paths(fname):
    paths = [line.strip() for line in open(fname).readlines()]
    if not paths:
        raise ValueError("Make sure the file is not empty!")
    return paths


def read_vocab(vocab_path):
    with open(vocab_path) as fin:
        words = [line.split(" ")[0] for line in fin]
    return words


def save_vocab(vocab_path, vocab):
    with open(vocab_path, "w") as fout:
        fout.write("\n".join(
            map(lambda x: "%s %d".join(x), vocab.most_common())))
