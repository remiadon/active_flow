import fire
from skmine.itemsets import SLIM
from skmine.datasets import fimi
import pandas as pd

import signal
import sys

import fire
from rich import print
from rich.console import Console

import os

import joblib

def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Base(object):
    def __init__(self, **kwargs):
        self.model = None
        self.data = None

    def create_model(self, output_path: str, klass_path: str, **kwargs):
        if not output_path.endswith(".pkl"):
            raise ValueError("Please use the `.pkl` file format to save you model")
        klass = my_import(klass_path)
        self.model = klass(**kwargs)
        return self  # TODO: return the model itself ? or inherit every function from it, like in dask-ml MetaEstimators

    def load_model(self, path: str):  # TODO get or create
        if self.model is not None:
            raise ValueError("only one model per run allowed")
        path = os.path.expanduser(path)
        self.model = joblib.load(path)
        return self

    def get_model(self, path: str, klass_path: str=""):
        """ get or create the model"""
        path = os.path.expanduser(path)
        if os.path.exists(path):
            return self.load_model(path)
        return self.create_model(klass_path, output_path=path)

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("model should be created or loaded before")
        joblib.dump(self.model, path)

    def load_data(self, path, *args, **kwargs):    # TODO fire pandas function directly ??
        self.data = pd.read_csv(path, *args, **kwargs)
        return self

    def fetch_data(self, fn, *args, **kwargs):    # TODO fire pandas function directly ??
        fetch_fn = my_import(fn)
        self.data = fetch_fn(*args, **kwargs)
        return self


class PatternMining(Base):
    def __init__(self, *args, save_on_quit=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.console = Console(record=True, stderr=True)

        if save_on_quit:
            def signal_handler(sig, frame):
                print("dumping model at slim.pkl")
                joblib.dump(self.model, "slim.pkl")
                sys.exit(0)
            signal.signal(signal.SIGINT, signal_handler)

    def active(self):
        slim = self.model
        D = self.data
        slim._prefit(D)
        stack = set()  # already seen candidates
        while True:
            cands = slim.generate_candidates(stack=stack)
            for cand, tids in cands:
                data_size, model_size, update_d, prune_set = slim.evaluate(cand)
                diff = (slim.model_size_ + slim.data_size_) - (data_size + model_size)
                include = self.console.input(f"""
                    Do we include candidate {cand} into our pattern set ?\n
                    estimated MDL diff : {diff}\n
                    Answer by `y`|`yes` or `n`|`no`
                """)
                if include.lower() in ("yes", "y"):
                    slim.codetable_.update(update_d)
                    slim.data_size_ = data_size
                    slim.model_size_ = model_size


if __name__ == '__main__':
  fire.Fire(dict(
      pattern_mining=PatternMining,
      read_csv=pd.read_csv,
  ))
