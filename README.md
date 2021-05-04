# Active Flow, a CLI for active learning and interactive data mining

## key features
- simple syntax, build entire pipelines in a single line of code
- dynamic loading of your favourite libraries (you can use what you have access to)
- create a model from scratch, or load an existing one
- interactive user input via CLI

# features to come
- more sampling strategies
- load/save models from MLFlow


# Example, a simple linear regression on the titanic dataset
```bash
python -m aflow.active --name=log_reg - create_model  lin_reg.pkl sklearn.linear_model.LogisticRegression - read_csv titanic_train.csv --usecols="[Age,Fare]" - loop uncertainty --batch_size=2 --n_iter=1
```
![2021-05-04 17 45 08](https://user-images.githubusercontent.com/2931080/117031794-37f04d80-ad01-11eb-9f43-e84e6362449d.gif)
