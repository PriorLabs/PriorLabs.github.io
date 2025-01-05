# Installation

To install TabPFN, please use the notebooks provided in this review. After review we are going to release our models as a python package on the python repository pypi.

## Example

A simple way to get started with TabPFN using our sklearn interface is demonstrated below. This example shows how to train a classifier on the breast cancer dataset and evaluate its accuracy.

```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize classifier
classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

# Train classifier
classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
```
This example demonstrates the basic workflow of training and predicting with TabPFN models. For more advanced usage, including handling of categorical data, please refer to the Advanced Usage section.
