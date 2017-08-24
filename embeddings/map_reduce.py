import multiprocessing
import time

from ast2vec.pickleable_logger import PickleableLogger
from modelforge.progress_bar import progress_bar


class MapReduce(PickleableLogger):
    def __init__(self, log_level, num_processes):
        super(MapReduce, self).__init__(log_level=log_level)
        self.num_processes = num_processes

    def parallelize(self, tasks, process_queue_in, process_queue_out):
        queue_in = multiprocessing.Manager().Queue()
        queue_out = multiprocessing.Manager().Queue(100)
        processes = [multiprocessing.Process(target=process_queue_in,
                                             args=(self, queue_in, queue_out))
                     for i in range(self.num_processes)]
        for p in processes:
            p.start()
        for t in tasks:
            queue_in.put(t)
        for _ in processes:
            queue_in.put(None)
        failures = process_queue_out(self, len(tasks), queue_out)
        for p in processes:
            p.join()
        self._log.info("Finished, %d failed tasks", failures)
        return len(tasks) - failures

    @staticmethod
    def read_vocab(vocab_path):
        with open(vocab_path) as fin:
            words = [line.split(" ")[0] for line in fin]
        return words

    @staticmethod
    def save_vocab(vocab_path, vocab):
        with open(vocab_path, "w") as fout:
            fout.write("\n".join(
                map(lambda x: "%s %d".join(x), vocab.most_common())))

    @staticmethod
    def wrap_queue_in(func):
        def wrapper(self, queue_in, queue_out):
            while True:
                item = queue_in.get()
                if item is None:
                    break
                try:
                    queue_out.put(func(self, item))
                except:
                    self._log.exception("%s failed", item)
                    queue_out.put(None)
        return wrapper

    @staticmethod
    def wrap_queue_out(func):
        def wrapper(self, n_tasks, queue_out):
            failures = 0
            start = time.time()

            for i in range(n_tasks):
                result = queue_out.get()
                if i % 1000 == 0:
                    print(i, time.time() - start)
                if result is None:
                    failures += 1
                    continue
                func(result)

            return failures
        return wrapper

    def _get_log_name(self):
        return "MapReduce"
