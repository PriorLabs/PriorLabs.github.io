# Classification

TabPFN provides a powerful interface for handling classification tasks on tabular data. The `TabPFNClassifier` class can be used for binary and multi-class classification problems.

## Example

Below is an example of how to use `TabPFNClassifier` for a multi-class classification task:

```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize a classifier
clf = TabPFNClassifier(fit_at_predict_time=True)
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities, multi_class="ovr"))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

## Example with AutoTabPFNClassifier

!!! abstract
	
	The AutoTabPFNClassifier and AutoTabPFNRegressor automatically run a hyperparameter search and build an ensemble of strong hyperparameters. You can control the runtime using ´max_time´ and need to make no further adjustments to get best results.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn.scripts.estimator.post_hoc_ensembles import AutoTabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Post-hoc ensemble (PHE) variant of TabPFN (we denote this as AutoTabPFN in our code)
clf = AutoTabPFNClassifier(device="auto", max_time=30)
clf.fit(X_train, y_train)

prediction_probabilities = clf.predict_proba(X_test)
predictions = np.argmax(prediction_probabilities, axis=1)  # Get labels from prediction_probabilities

print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))
print("Accuracy", accuracy_score(y_test, predictions))
```