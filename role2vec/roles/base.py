import os

from sklearn.externals import joblib

from ast2vec.token_parser import TokenParser
from map_reduce import MapReduce
from utils import read_embeddings

ROLES_MODELS = dict()


def register_roles_model(cls):
    """
    Check some conventions for class declaration and add it to ROLES_MODELS.

    :param cls: Class for roles prediction.
    """
    base = "Roles"
    assert issubclass(cls, RolesBase), "Must be a subclass of RolesBase."
    assert cls.__name__.startswith(base), "Make sure to start your class name with %s." % (base, )
    ROLES_MODELS[cls.__name__[len(base):].lower()] = cls

    return cls


class RolesBase(MapReduce):
    """
    Base class for roles prediction.
    """

    def __init__(self, log_level: str, num_processes: int, emb_path: str):
        """
        :param log_level: Log level of RolesBase.
        :param num_processes: Number of running processes. There's always one additional process
                              for reducing data.
        :param emb_path: Path to stored roles embeddings.
        """
        super(RolesBase, self).__init__(log_level=log_level, num_processes=num_processes)
        self.emb, self.roles = read_embeddings(emb_path)
        self.model = None
        self.token_parser = TokenParser()

    def save(self, model_path: str) -> None:
        """
        Store trained model on disk.

        :param model_path: Path for storing trained model.
        """
        if self.model is None:
            raise ValueError("Model is empty.")
        self._log.info("Saving model to %s.", model_path)
        joblib.dump(self.model, model_path)

    def load(self, model_path: str) -> None:
        """
        Load trained model from disk.

        :param model_path: Path to trained model.
        """
        if not os.path.exists(model_path):
            raise ValueError("Provided path to model doesn't exist: %s", model_path)
        self.model = joblib.load(model_path)

    def train(self, fname: str) -> None:
        """
        Train model.

        :param fname: Path to train file with filepaths to stored UASTs.
        """
        raise NotImplementedError

    def test(self, fname: str) -> None:
        """
        Test model.

        :param fname: Path to test file with filepaths to stored UASTs.
        """
        raise NotImplementedError


def roles_entry(args):
    RolesModel = ROLES_MODELS[args.algorithm]
    rm = RolesModel(args.log_level, args.processes, args.embeddings)

    if args.train:
        rm.train(args.train)
        rm.save(args.model)
    else:
        rm.load(args.model)

    if args.test:
        rm.test(args.test)
