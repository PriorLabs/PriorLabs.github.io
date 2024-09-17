# Regression

TabPFN can also be applied to regression tasks using the `TabPFNRegressor` class. This allows for predictive modeling of continuous outcomes.

## Example

An example usage of `TabPFNRegressor` is shown below:

```python
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize a regressor
reg = TabPFNRegressor()
reg.fit(X_train, y_train)

# Predict
predictions = reg.predict(X_test)

print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

# Predict distribution for regression
full_predictions = reg.predict_full(X_test)  # returns a dict with predictions for mean, median, mode, and quantiles.
print("R-squared with median instead of mean predictions:", r2_score(y_test, full_predictions["median"]))
```
This example demonstrates how to train and evaluate a regression model. For more details on TabPFNRegressor and its parameters, refer to the API Reference section.

## Example with AutoTabPFNRegressor

!!! abstract
	
	The AutoTabPFNClassifier and AutoTabPFNRegressor automatically run a hyperparameter search and build an ensemble of strong hyperparameters. You can control the runtime using ´max_time´ and need to make no further adjustments to get best results.

```python
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn.scripts.estimator.post_hoc_ensembles import AutoTabPFNRegressor

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize a regressor
reg = AutoTabPFNRegressor(device="auto", max_time=30)
reg.fit(X_train, y_train)

# Predict
predictions = reg.predict(X_test)

print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))
```

## Example HPO for TabPFN

!!! abstract
	
	You can use a HPO tool, like [Optuna](https://optuna.org/), to optimize the hyerparameters of TabPFN yourself or integrate it in your personal pipeline.
    [Install Optuna](https://optuna.org/#installation) to run the example below. 

```python
from pathlib import Path

import optuna  # Install optuna first
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score

from tabpfn import TabPFNRegressor
from tabpfn.scripts.estimator.base import local_model_path
from tabpfn.scripts.estimator.hpo.search_space import enumerate_preprocess_transforms

preprocessing_options = enumerate_preprocess_transforms()


def objective(trial):
    # -- Load toy data
    iris = load_diabetes()
    X, y = iris.data, iris.target

    # -- Suggest/sample hyperparameters to evaluate
    # See TabPFNClassifier parameters for potential options.
    # See tabpfn.scripts.estimator.hpo.search_space.get_param_grid_hyperopt for the search space we used in the paper.
    tabpfn_pretrained_model = trial.suggest_categorical(
        "tabpfn_pretrained_model",
        [
            Path(local_model_path) / "model_hans_regression.ckpt",
            Path(local_model_path) / "model_hans_regression_2noar4o2.ckpt",
            Path(local_model_path) / "model_hans_regression_5wof9ojf.ckpt",
            Path(local_model_path) / "model_hans_regression_09gpqh39.ckpt",
            Path(local_model_path) / "model_hans_regression_wyl4o83o.ckpt",
        ],
    )
    n_estimators = trial.suggest_categorical("n_estimators", [1, 4, 8])
    preprocessing = trial.suggest_categorical("preprocessing", preprocessing_options)
    softmax_temperature = trial.suggest_categorical(
        "softmax_temperature",
        [
            0.75,
            0.8,
            0.9,
            0.95,
            1.0,
        ],
    )

    # -- Initialize model
    classifier_obj = TabPFNRegressor(
        model_path=tabpfn_pretrained_model,
        n_estimators=n_estimators,
        preprocess_transforms=preprocessing,
        softmax_temperature=softmax_temperature,
    )
    # -- Evaluate the model
    score = cross_val_score(classifier_obj, X, y, n_jobs=1, cv=3)
    return score.mean()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best Parameters:", study.best_params)
    print("Best Score:", study.best_value)

```