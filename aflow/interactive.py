import fire
import pandas as pd
import datetime as dt

import signal
import sys

import fire
from rich import print
from rich.console import Console

import os

import joblib


class Interactive(Base):
    def __init__(self, *args, name=None, save_on_quit=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.console = Console(record=True, stderr=True)

        if save_on_quit:
            def signal_handler(sig, frame):
                name = self.name + "pkl"
                print(f"dumping model at {name}")
                joblib.dump(self.model, name)
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
  fire.Fire(Interactive)
