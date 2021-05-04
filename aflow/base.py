"""
Base definitions to interact with different Machine Learning / Data Mining Libraries

Defitions are really broad, on purpose. We don't want to import anything.
"""
from typing import Callable
import datetime as dt
import joblib
import pandas as pd

from rich import print
import sys

def dyn_import(name):
    components = name.split('.')
    mod = __import__(".".join(components[:-1]))
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def check_requirements(*reqs):
    def inner(fn: Callable):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            for req in reqs:
                assert hasattr(result, req)
            return result
        return wrapper
    return inner

def _check_is_fitted(estimator, X):
    """
    generic check for estimator is fitted
    without calling sklearn.utils.validation.check_is_fitted
    """  # TODO add fit_one and transform_one from `river ml`
    fn = None
    sample = X[:5]
    if hasattr(estimator, "predict"):
        fn = estimator.predict
    elif hasattr(estimator, "transform"):
        fn = estimator.transform
    else:
        raise TypeError(f"{estimator} is not an estimator")
    try:
        fn(sample)
    except:
        print(f"{estimator} not fitted")
        return False
    return True


class Base(object):
    def __init__(self, name=None, auto_save=True, **kwargs):
        self.model = None
        self.data = None
        if name is None:
            name = dt.datetime.now()
            name = name.strftime("%d-%b-%Y_%H:%M")
        self.name = name

        self.auto_save = auto_save

    @check_requirements("fit")
    def create_model(self, output_path: str, klass: str, **kwargs):
        if not output_path.endswith(".pkl"):
            raise ValueError("Please use the `.pkl` file format to save you model")
        klass = dyn_import(klass)
        self.model = klass(**kwargs)
        return self  # TODO: return the model itself ? or inherit every function from it, like in dask-ml MetaEstimators

    @check_requirements("fit")
    def load_model(self, path: str):  # TODO get or create
        if self.model is not None:
            raise ValueError("only one model per run allowed")
        _check_is_fitted(self.model)
        path = os.path.expanduser(path)
        self.model = joblib.load(path)
        return self

    def fit(self, X, y):
        if _check_is_fitted(self.model, X):
            return self.model.partial_fit(X, y)
        return self.model.fit(X, y)  # fallback


    def read_csv(self, path, *args, **kwargs):    # TODO fire pandas function directly ??
        __doc__ = pd.read_csv.__doc__
        self.data = pd.read_csv(path, *args, **kwargs)
        return self

    def fetch_data(self, fn, *args, **kwargs):    # TODO fire pandas function directly ??
        fetch_fn = my_import(fn)
        self.data = fetch_fn(*args, **kwargs)
        return self

    def __del__(self, *args, **kwargs):
        if self.auto_save and self.model is not None:
            name = self.name + ".pkl"
            print(f"dumping model @ {name} :thumbs_up:")
            joblib.dump(self.model, name)
        del self
