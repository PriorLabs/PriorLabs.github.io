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
clf = TabPFNClassifier()
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

## Example HPO for TabPFN

!!! abstract
	
	You can use a HPO tool, like [Optuna](https://optuna.org/), to optimize the hyerparameters of TabPFN yourself or integrate it in your personal pipeline.
    [Install Optuna](https://optuna.org/#installation) to run the example below. 

```python
from pathlib import Path

import optuna  # Install optuna first
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from tabpfn import TabPFNClassifier
from tabpfn.scripts.estimator.base import local_model_path
from tabpfn.scripts.estimator.hpo.search_space import enumerate_preprocess_transforms

preprocessing_options = enumerate_preprocess_transforms()


def objective(trial):
    # -- Load toy data
    iris = load_iris()
    X, y = iris.data, iris.target

    # -- Suggest/sample hyperparameters to evaluate
    # See TabPFNClassifier parameters for potential options.
    # See tabpfn.scripts.estimator.hpo.search_space.get_param_grid_hyperopt for the search space we used in the paper.
    tabpfn_pretrained_model = trial.suggest_categorical(
        "tabpfn_pretrained_model",
        [
            Path(local_model_path) / "model_hans_classification.ckpt",
            Path(local_model_path) / "model_hans_classification_gn2p4bpt.ckpt",
            Path(local_model_path) / "model_hans_classification_llderlii.ckpt",
            Path(local_model_path) / "model_hans_classification_od3j1g5m.ckpt",
            Path(local_model_path) / "model_hans_classification_vutqq28w.ckpt",
            Path(local_model_path) / "model_hans_classification_znskzxi4.ckpt",
        ],
    )
    n_estimators = trial.suggest_categorical("n_estimators", [1, 4, 8])
    preprocessing = trial.suggest_categorical("preprocessing", preprocessing_options)
    softmax_temperature = trial.suggest_categorical(
        "softmax_temperature",
        [
            0.75,
            0.8,
            0.9,
            0.95,
            1.0,
        ],
    )

    # -- Initialize model
    classifier_obj = TabPFNClassifier(
        model_path=tabpfn_pretrained_model,
        n_estimators=n_estimators,
        preprocess_transforms=preprocessing,
        softmax_temperature=softmax_temperature,
    )
    # -- Evaluate the model
    score = cross_val_score(classifier_obj, X, y, n_jobs=1, cv=3)
    return score.mean()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best Parameters:", study.best_params)
    print("Best Score:", study.best_value)
```