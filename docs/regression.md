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
reg = TabPFNRegressor(device="auto")
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