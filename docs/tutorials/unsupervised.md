# Unsupervised functionalities

```python
from tabpfn.scripts.estimator import TabPFNUnsupervisedModel, TabPFNClassifier, TabPFNRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
model_unsupervised = TabPFNUnsupervisedModel(TabPFNClassifier(), TabPFNRegressor())

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

model_unsupervised.fit(X_train, y_train)
embeddings = model_unsupervised.get_embeddings(X_test)
X_outliers = model_unsupervised.outliers(X_test)
X_synthetic = model_unsupervised.generate_synthetic_data(n_samples=100)
```