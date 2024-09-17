# Unsupervised functionalities

```python
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tabpfn.scripts.estimator import TabPFNClassifier, TabPFNRegressor, TabPFNUnsupervisedModel

# Load and split the Iris dataset
X, y = load_iris(return_X_y=True)
X, y = torch.tensor(X), torch.tensor(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Load Model
model_unsupervised = TabPFNUnsupervisedModel(TabPFNClassifier(), TabPFNRegressor())
model_unsupervised.fit(X_train, y_train)

# Unsupervised functions
embeddings = model_unsupervised.get_embeddings(X_test)
X_outliers = model_unsupervised.outliers(X_test)
X_synthetic = model_unsupervised.generate_synthetic_data(n_samples=100)
```