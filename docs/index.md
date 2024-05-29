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

=== "TabPFN"

    ```
    torch>=2.1 (Includes CUDA support in version 2.1 and later)
    scikit-learn>=1.4.2
    tqdm>=4.66.
    numpy>=1.21.2
    hyperopt==0.2.7 (Note: Earlier versions fail with numpy number generator change)
    pre-commit>=3.3.3
    einops>=0.6.0
    scipy>=1.8.0
    torchmetrics==1.2.0
    pytest>=7.1.3
    pandas[plot,output_formatting]>=2.0.3,<2.2 (Note: Version 2.2 has a bug with multi-index tables (https://github.com/pandas-dev/pandas/issues/57663), recheck when fixed)
    pyyaml>=6.0.1
    kditransform>=0.2.0
    ```

=== "TabPFN and Baselines"

    ```
    torch>=2.1 (Includes CUDA support in version 2.1 and later)
    scikit-learn>=1.4.2
    tqdm>=4.66.
    numpy>=1.21.2
    hyperopt==0.2.7 (Note: Earlier versions fail with numpy number generator change)
    pre-commit>=3.3.3
    einops>=0.6.0
    scipy>=1.8.0
    torchmetrics==1.2.0
    pytest>=7.1.3
    pandas[plot,output_formatting]>=2.0.3,<2.2 (Note: Version 2.2 has a bug with multi-index tables (https://github.com/pandas-dev/pandas/issues/57663), recheck when fixed)
    pyyaml>=6.0.1
    kditransform>=0.2.0
    seaborn==0.12.2
    openml==0.14.1
    numba>=0.58.1
    shap>=0.44.1
    
    # Baselines
    lightgbm==3.3.5
    xgboost>=2.0.0
    catboost>=1.1.1
    #auto-sklearn==0.14.5
    #autogluon==0.4.0
    
    # -- Quantile Baseline
    quantile-forest==1.2.4
    ```

For GPU usage CUDA 12.1 has been tested.

### Non-Standard Hardware
GPU: A CUDA-enabled GPU is recommended for optimal performance, though the software can also run on a CPU.

## Installation

To install our software, we use pip the python package installer in combination with Git for code-management. Please find the code for installation via the private “linktree” shared with you, that also contains the private access tokens to the code. An installation typically takes 5 minutes in a setup python environment. 



## Example usage

=== "Classification"

    ```python
    import numpy as np
    import sklearn
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    from tabpfn import TabPFNClassifier
    
    # Create a classifier
    clf = TabPFNClassifier(fit_at_predict_time=True)
    
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)
    y_eval = np.argmax(preds, axis=1)
    
    print('ROC AUC: ', sklearn.metrics.roc_auc_score(y_test, preds, multi_class='ovr'), 'Accuracy', sklearn.metrics.accuracy_score(y_test, y_eval))
    ```

=== "Regression"

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

## Expected Output
Our models follow the interfaces provided by sklearn, so you can expect the same output as you would from sklearn models.
TabPFNClassifier will return a numpy array of shape `(n_samples, n_classes)` with the probabilities of each class, while
TabPFNRegressor will return a numpy array of shape `(n_samples,)` with the predicted values. For more detailed documentation
please check the technical documentation of [scripts.estimator.TabPFNClassifier.predict_proba](https://priorlabs.github.io/api/tabpfn_classifier/#scripts.estimator.TabPFNClassifier.predict_proba).

## Expected Runtime
The runtime of the model is dependent on the number of estimators and the size of the dataset. For a dataset of 1000
samples and 4 features, the runtime on GPU is typically less than 1 second. For a dataset of 10000 samples and 4 features, the
runtime on GPU is typically less than 10 seconds.