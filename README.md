# Active Flow, a CLI for active learning and interactive data mining

## key features
- simple syntax, build entire pipelines in a single line of code
- dynamic loading of your favourite libraries (you can use what you have access to)
- interactive user input via CLI

# features to come
- sampling for active learning (use dask ?)
- load/save models from MLFlow


# Example
```bash
python pattern_mining.py pattern_mining - load_model slim.pkl - fetch_data skmine.datasets.fimi.fetch_mushroom - active
```