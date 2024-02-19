# Installation

To install TabPFN, you can simply use pip. The basic installation is suitable for most users who are interested in applying TabPFN models to their tabular data.

```bash
pip install tabpfn
For users interested in a more comprehensive setup, including the ability to train models, evaluate them as done in our paper, and use baselines, the full installation is recommended:
```

```bash
pip install tabpfn[full]
```
Note: To use AutoGluon and Auto-sklearn baselines, please create separate environments and install autosklearn==0.14.5 and autogluon==0.4.0 respectively. Installing them in the same environment as TabPFN may not be possible due to dependency conflicts.

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
