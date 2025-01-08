# Regression

TabPFN can also be applied to regression tasks using the `TabPFNRegressor` class. This allows for predictive modeling of continuous outcomes.

## Example

An example usage of `TabPFNRegressor` is shown below:

=== "Python API Client (No GPU, Online)"

	```python
	from tabpfn_client import TabPFNRegressor
	from sklearn.datasets import load_diabetes
	from sklearn.model_selection import train_test_split
	import numpy as np
	import sklearn
	
	reg = TabPFNRegressor(device='auto')
	X, y = load_diabetes(return_X_y=True)
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	reg.fit(X_train, y_train)
	preds = reg.predict(X_test)
	
	print('Mean Squared Error (MSE): ', sklearn.metrics.mean_squared_error(y_test, preds))
	print('Mean Absolute Error (MAE): ', sklearn.metrics.mean_absolute_error(y_test, preds))
	print('R-squared (R^2): ', sklearn.metrics.r2_score(y_test, preds))
	```
	
=== "Python Local (GPU)"

	```python
	from tabpfn import TabPFNRegressor
	from sklearn.datasets import load_diabetes
	from sklearn.model_selection import train_test_split
	import numpy as np
	import sklearn
	
	reg = TabPFNRegressor(device='auto')
	X, y = load_diabetes(return_X_y=True)
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	reg.fit(X_train, y_train)
	preds = reg.predict(X_test)
	
	print('Mean Squared Error (MSE): ', sklearn.metrics.mean_squared_error(y_test, preds))
	print('Mean Absolute Error (MAE): ', sklearn.metrics.mean_absolute_error(y_test, preds))
	print('R-squared (R^2): ', sklearn.metrics.r2_score(y_test, preds))
	```
	
This example demonstrates how to train and evaluate a regression model. For more details on TabPFNRegressor and its parameters, refer to the API Reference section.

## Example with AutoTabPFNRegressor


!!! abstract
	
	AutoTabPFNRegressor yields the most accurate predictions for TabPFN and is recommended for most use cases.
    The AutoTabPFNClassifier and AutoTabPFNRegressor automatically run a hyperparameter search and build an ensemble of strong hyperparameters.
    You can control the runtime using ´max_time´ and need to make no further adjustments to get best results.

```python
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn

reg = AutoTabPFNRegressor(max_time=30) # runs for 30 seconds
X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)

print('Mean Squared Error (MSE): ', sklearn.metrics.mean_squared_error(y_test, preds))
print('Mean Absolute Error (MAE): ', sklearn.metrics.mean_absolute_error(y_test, preds))
print('R-squared (R^2): ', sklearn.metrics.r2_score(y_test, preds))
```