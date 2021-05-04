# a CLI for active learning & interactive data mining
<img src="https://user-images.githubusercontent.com/2931080/117039599-3e82c300-ad09-11eb-9749-189730d3b196.png" alt="drawing" width="400"/>


## key features
- simple syntax, build entire pipelines in a single line of code
- dynamic loading of your favourite libraries (you can use what you have access to)
- create a model from scratch, or load an existing one
- interactive user input via CLI

## features to come
- more sampling strategies
- load/save models from MLFlow


## Example with a simple logitic regression on the titanic dataset
![2021-05-04 18 21 54](https://user-images.githubusercontent.com/2931080/117036759-13e33b00-ad06-11eb-8bb8-709524637f52.gif)


## Documentation
active_flow uses [python fire](https://github.com/google/python-fire) internally, so you can call eg.
```bash
python -m aflow.active --help
```
