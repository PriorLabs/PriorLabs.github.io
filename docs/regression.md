# Regression

TabPFN can also be applied to regression tasks using the `TabPFNRegressor` class. This allows for predictive modeling of continuous outcomes.

## Example

An example usage of `TabPFNRegressor` is shown below:

```python
from tabpfn import TabPFNRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load Boston housing dataset
X, y = load_boston(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train regressor
regressor = TabPFNRegressor(device='cuda', N_ensemble_configurations=10)
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)
print('Test RMSE:', mean_squared_error(y_test, y_pred, squared=False))
```
This example demonstrates how to train and evaluate a regression model. For more details on TabPFNRegressor and its parameters, refer to the API Reference section.