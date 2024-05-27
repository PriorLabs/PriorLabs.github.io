# Highly Accurate Predictions for Small Data With the Tabular Foundation Model TabPFN (beta)

This page contains usage examples and installation instructions of TabPFN. Please find additional instructions on our Classifiers and Regressors on the respective subpages (menu left). You will also find an in-depth technical documentation of our source code with a documentation of our software interfaces on the left hand side.

Please do not share this review version.

## System Requirements

### Software Dependencies and Operating Systems
Python: Version >= 3.9
Operating Systems: The software has been tested on major operating systems including:
- Ubuntu 20.04, 22.04
- Windows 10, 11
- macOS 11.0 (Big Sur) and later
Git Version 2 or later ([https://git-scm.com/](https://git-scm.com/))

### Software Dependencies (as specified in `requirements.txt`):
torch>=2.1
Includes CUDA support in version 2.1 and later
scikit-learn>=1.4.2
tqdm>=4.66.1
numpy>=1.21.2
hyperopt==0.2.7
Note: Earlier versions fail with numpy number generator change
pre-commit>=3.3.3
einops>=0.6.0
scipy>=1.8.0
torchmetrics==1.2.0
pytest>=7.1.3
pandas[plot,output_formatting]>=2.0.3,<2.2
Note: Version 2.2 has a bug with multi-index tables (https://github.com/pandas-dev/pandas/issues/57663), recheck when fixed
pyyaml>=6.0.1
kditransform>=0.2.0

For GPU usage CUDA 12.1 has been tested.

### Non-Standard Hardware
GPU: A CUDA-enabled GPU is recommended for optimal performance, though the software can also run on a CPU.

## Installation

To install our software, we use pip the python package installer in combination with Git for code-management. Please find the code for installation via the private “linktree” shared with you, that also contains the private access tokens to the code.

## Example usage
```python
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier
 
# Create a classifier
# The model will use a CUDA-enabled GPU if available, otherwise it will use the CPU.
# The most important parameters:
#   - The default is to `fit_at_predict_time` which means that the model will be fit at the time of prediction, this is useful for large datasets and when you only want to predict once.
#     If you want to predict multiple times for the same training set, set `fit_at_predict_time=False` and the model will be fit at the time of `fit` and save its state for later.
#   - You can also set the number of `n_estimators`, this is the easiest way to control the trade-off between speed and accuracy.
clf = TabPFNClassifier(fit_at_predict_time=True)

X, y = load_iris(return_X_y=True)
feature_names = load_iris()['feature_names']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf.fit(X_train, y_train)

preds = clf.predict_proba(X_test)  # <- all the compute happens in here
y_eval = np.argmax(preds, axis=1)

print('ROC AUC: ',  sklearn.metrics.roc_auc_score(y_test, preds, multi_class='ovr'), 'Accuracy', sklearn.metrics.accuracy_score(y_test, y_eval))
```


## Expected Output
Our models follow the interfaces provided by sklearn, so you can expect the same output as you would from sklearn models.
TabPFNClassifier will return a numpy array of shape `(n_samples, n_classes)` with the probabilities of each class, while
TabPFNRegressor will return a numpy array of shape `(n_samples,)` with the predicted values. For more detailed documentation
please check the technical documentation of [scripts.estimator.TabPFNClassifier.predict_proba](https://priorlabs.github.io/api/tabpfn_classifier/#scripts.estimator.TabPFNClassifier.predict_proba).

## Expected Runtime
The runtime of the model is dependent on the number of estimators and the size of the dataset. For a dataset of 1000
samples and 4 features, the runtime on GPU is typically less than 1 second. For a dataset of 10000 samples and 4 features, the
runtime on GPU is typically less than 10 seconds, .