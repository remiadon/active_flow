from .base import Base
from .sampling import uncertainty
import pandas as pd

from rich import print
from rich.console import Console
import fire

import joblib
import sys

import signal

class Active(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console = Console(record=True, stderr=True)
        # load .classes_ at init, and load when load_model

        print(f"Active meta learner instanciated with {args}, {kwargs}")

    def loop(self, name, batch_size=10, n_iter:int = 2, *args):
        if name not in ("uncertainty", ):  # TODO : allow functions from modAL ?
            raise ValueError("only uncertainty sampling is supported for now")

        for i in range(n_iter):
            cands = uncertainty(self.model, self.data)
            batch_indices = cands.argsort()[-batch_size:]
            batch = self.data.iloc[batch_indices].copy()  # beware, no `.iloc` in dask

            y = list()
            for x in batch.to_dict(orient="records"):
                y_single = self.console.input(f"""Please provide a label for {x}\n""")
                if not y_single:
                    raise ValueError("you must answer the question")
                y.append(y_single)
            self.fit(batch, y)

            batch.loc[:, "_target"] = y
            print(f"fitted model on data: \n{batch}\n")



if __name__ == '__main__':
  fire.Fire(Active)
