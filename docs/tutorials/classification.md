# Classification

TabPFN provides a powerful interface for handling classification tasks on tabular data. The `TabPFNClassifier` class can be used for binary and multi-class classification problems.

## Example

Below is an example of how to use `TabPFNClassifier` for a multi-class classification task:

=== "Python API Client (No GPU, Online)"

	```python
	from tabpfn_client import TabPFNClassifier
	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	
	# Load the Iris dataset
	X, y = load_iris(return_X_y=True)
	
	# Split data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Initialize and train classifier
	classifier = TabPFNClassifier(device='cuda', N_ensemble_configurations=10)
	classifier.fit(X_train, y_train)
	
	# Evaluate
	y_pred = classifier.predict(X_test)
	print('Test Accuracy:', accuracy_score(y_test, y_pred))
	```
=== "Python Local (GPU)"

	```python
	from tabpfn import TabPFNClassifier
	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	
	# Load the Iris dataset
	X, y = load_iris(return_X_y=True)
	
	# Split data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Initialize and train classifier
	classifier = TabPFNClassifier(device='cuda', N_ensemble_configurations=10)
	classifier.fit(X_train, y_train)
	
	# Evaluate
	y_pred = classifier.predict(X_test)
	print('Test Accuracy:', accuracy_score(y_test, y_pred))
	```


## Example with AutoTabPFNClassifier

!!! abstract
	
	AutoTabPFNClassifier yields the most accurate predictions for TabPFN and is recommended for most use cases.
    The AutoTabPFNClassifier and AutoTabPFNRegressor automatically run a hyperparameter search and build an ensemble of strong hyperparameters.
    You can control the runtime using ´max_time´ and need to make no further adjustments to get best results.

```python
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

# we refer to the PHE variant of TabPFN as AutoTabPFN in the code
clf = AutoTabPFNClassifier(device='auto', max_time=30)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf.fit(X_train, y_train)

preds = clf.predict_proba(X_test)
y_eval = np.argmax(preds, axis=1)

print('ROC AUC: ',  sklearn.metrics.roc_auc_score(y_test, preds[:,1], multi_class='ovr'), 'Accuracy', sklearn.metrics.accuracy_score(y_test, y_eval))
```