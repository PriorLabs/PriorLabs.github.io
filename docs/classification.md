# Classification

TabPFN provides a powerful interface for handling classification tasks on tabular data. The `TabPFNClassifier` class can be used for binary and multi-class classification problems.

## Example

Below is an example of how to use `TabPFNClassifier` for a multi-class classification task:

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