import multiprocessing
import time
from typing import List

from ast2vec.pickleable_logger import PickleableLogger


class MapReduce(PickleableLogger):
    """
    Base class for parallel data processign. Creates a pool of workers for data mangling and
    reduces data in the main process.
    """

    def __init__(self, log_level: str, num_processes: int, queue_lim: int=100):
        """
        :param log_level: Log level of MapReduce.
        :param num_processes: Number of running processes. There's always one additional process
                              for reducing data.
        :param queue_lim: Maximum number of results in queue for reducing.
        """
        super(MapReduce, self).__init__(log_level=log_level)
        self.num_processes = num_processes
        self.queue_lim = queue_lim

    def parallelize(self, tasks: List[str], process_queue_in, process_queue_out) -> int:
        """
        Process tasks in parallel.

        :param tasks: List of filenames.
        :param process_queue_in: Function for processing items from the task queue.
        :param process_queue_out: Function for processing items from the result queue.
        :return: Number of failed tasks.
        """
        queue_in = multiprocessing.Manager().Queue()
        queue_out = multiprocessing.Manager().Queue(self.queue_lim)
        processes = [multiprocessing.Process(target=process_queue_in,
                                             args=(self, queue_in, queue_out))
                     for i in range(self.num_processes)]
        n_tasks = len(tasks)
        start_time = time.time()

        self._log.info("Starting tasks.")
        for p in processes:
            p.start()
        for t in tasks:
            queue_in.put(t)
        for _ in processes:
            queue_in.put(None)

        failures = process_queue_out(self, n_tasks, queue_out)
        for p in processes:
            p.join()

        self._log.info("Finished %d/%d tasks in %.2f" %
                       (n_tasks - failures, n_tasks, time.time() - start_time))
        return len(tasks) - failures

    @staticmethod
    def wrap_queue_in(func):
        """
        Wrapper for automatic quering of tasks and storing results in the result queue.

        :param func: Function that can process a single task and accepts `self` as parameter.
        """
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
    def wrap_queue_out(freq: int=1000):
        """
        Wrapper for allowing parametrization.

        :param freq: Logs information every `freq` iterations.
        """
        def outer_wrapper(func):
            """
            Wrapper for automatic quering of results and reducing them.

            :param func: Function that can process a result and accepts `self` as parameter.
            """
            def wrapper(self, n_tasks, queue_out):
                failures = 0
                start = time.time()

                for i in range(n_tasks):
                    result = queue_out.get()
                    if (i + 1) % freq == 0:
                        self._log.info("Processed %d/%d in %.2f" %
                                       (i + 1, n_tasks, time.time() - start))
                    if result is None:
                        failures += 1
                        continue
                    func(self, result)

                self._log.info("Finished %d/%d in %.2f seconds" %
                               (i + 1, n_tasks, time.time() - start))
                return failures
            return wrapper
        return outer_wrapper

    def _get_log_name(self):
        return "MapReduce"
