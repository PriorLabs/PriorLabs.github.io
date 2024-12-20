# &nbsp;

<style>
.prior-labs-hero {
  position: relative;
  width: 100%;
  height: 00px;
  margin: 2rem 0;
  overflow: hidden;
  margin-top: -4rem;
}

.grid-container {
  display: grid;
  grid-template-columns: repeat(60, 1fr);
  grid-template-rows: repeat(26, 1fr);
  width: 100%;
  height: 100%;
  gap: 0.25%;
}

.square {
  background-color: rgba(96, 191, 129, 0.7);
  transition: background-color 0.5s ease;
  border-radius: 0px;
}

.square.active {
  background-color: rgba(96, 191, 129, 1);
}

.square.missing {
  background-color: transparent;
}

.hero-content {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  z-index: 10;
  padding: 2rem;
  border-radius: 8px;
}

.hero-title {
  font-size: 2.5rem !important;
  font-weight: 700 !important;
  color: rgb(96, 191, 129) !important;
  margin-bottom: 1rem !important;
  line-height: 1.2 !important;
}

.hero-subtitle {
  font-size: 1.2rem !important;
  color: rgb(96, 191, 129) !important;
  max-width: 600px;
  margin: 0 auto !important;
  line-height: 1.4 !important;
}
</style>

<div class="prior-labs-hero">
  <div class="grid-container">
    <!-- Squares will be generated by JavaScript -->
  </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', () => {
  const gridContainer = document.querySelector('.grid-container');
  const columns = 60;
  const rows = 25;
  const totalCells = columns * rows;

  // PriorLabs pattern (1 represents missing cell)
  const pattern = [
    "11111 11111 1 11111 11111    1       1    11111 11111",
    "1   1 1   1 1 1   1 1   1    1      111   1   1 1    ",
    "11111 11111 1 1   1 11111    1     11 11  11111 11111",
    "1     1 1   1 1   1 1 1      1    1111111 1   1     1",
    "1     1  11 1 11111 1  11    1111 1     1 11111 11111"
  ];

  // Create squares
  for (let i = 0; i < totalCells; i++) {
    const square = document.createElement('div');
    square.classList.add('square');
    gridContainer.appendChild(square);
  }

  // Apply PriorLabs pattern
  const squares = document.querySelectorAll('.square');
  const patternStartRow = 10;
  const patternStartCol = 3;
  
  pattern.forEach((row, rowIndex) => {
    for (let colIndex = 0; colIndex < row.length; colIndex++) {
      if (row[colIndex] === '1') {
        const index = (rowIndex + patternStartRow) * columns + (colIndex + patternStartCol);
        if (index < squares.length) {
          squares[index].classList.add('missing');
        }
      }
    }
  });

  // Animate squares
  setInterval(() => {
    squares.forEach(square => {
      if (!square.classList.contains('missing')) {
        if (Math.random() < 0.2) {
          square.classList.add('active');
        } else {
          square.classList.remove('active');
        }
      }
    });
  }, 1000);
});
</script>

PriorLabs is building breakthrough foundation models that understand spreadsheets and databases. While foundation models have transformed text and images, tabular data has remained largely untouched. We're tackling this opportunity with technology that could revolutionize how we approach scientific discovery, medical research, financial modeling, and business intelligence.

<!---
This page contains usage examples and installation instructions of TabPFN. Please find additional instructions on our Classifiers and Regressors on the respective subpages. An in-depth technical documentation of our software interfaces can be found in the [API Reference](api/tabpfn_classifier/)

## User Interface

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.43.0/gradio.js"></script>

<gradio-app src="https://noahho-tabpfn-client-gui.hf.space" theme_mode="light" eager="true" container="false">
</gradio-app>
-->

<!---
## Expected Output
Our models follow the interfaces provided by sklearn, so you can expect the same output as you would from sklearn models.
TabPFNClassifier will return a numpy array of shape `(n_samples, n_classes)` with the probabilities of each class, while
TabPFNRegressor will return a numpy array of shape `(n_samples,)` with the predicted values. For more detailed documentation
please check the technical documentation of [scripts.estimator.TabPFNClassifier.predict_proba](https://priorlabs.github.io/api/tabpfn_classifier/#scripts.estimator.TabPFNClassifier.predict_proba).

## Expected Runtime
The runtime of the model is dependent on the number of estimators and the size of the dataset. For a dataset of 1000
samples and 4 features, the runtime on GPU is typically less than 1 second. For a dataset of 10000 samples and 4 features, the
runtime on GPU is typically less than 10 seconds.
-->


## Why TabPFN

<div class="grid cards" markdown>

-   :material-speedometer:{ .lg .middle } **Rapid Training**

    ---

    TabPFN significantly reduces training time, outperforming traditional models tuned for hours in just a few seconds. For instance, it surpasses an ensemble of the strongest baselines in 2.8 seconds compared to 4 hours of tuning.

    [comment]: <> ([:octicons-arrow-right-24: Learn More](#))

-   :material-chart-line:{ .lg .middle } **Superior Accuracy**

    ---

    TabPFN consistently outperforms state-of-the-art methods like gradient-boosted decision trees (GBDTs) on datasets with up to 10,000 samples. It achieves higher accuracy and better performance metrics across a range of classification and regression tasks.

-   :material-shield-check:{ .lg .middle } **Robustness**

    ---

    The model demonstrates robustness to various dataset characteristics, including uninformative features, outliers, and missing values, maintaining high performance where other methods struggle.

-   :material-creation-outline:{ .lg .middle } **Generative Capabilities**

    ---

    As a generative transformer-based model, TabPFN can be fine-tuned for specific tasks, generate synthetic data, estimate densities, and learn reusable embeddings. This makes it versatile for various applications beyond standard prediction tasks.

-   :material-code-tags-check:{ .lg .middle } **Sklearn Interface**

    ---

    TabPFN follows the interfaces provided by scikit-learn, making it easy to integrate into existing workflows and utilize familiar functions for fitting, predicting, and evaluating models.

-   :material-file-excel-box:{ .lg .middle } **Minimal Preprocessing**

    ---

    The model handles various types of raw data, including missing values and categorical variables, with minimal preprocessing. This reduces the burden on users to perform extensive data preparation.

</div>

## TabPFN Integrations

<div class="grid cards" markdown>

-   :material-cloud-check:{ .lg .middle } **API Client**

    ---

    The fastest way to get started with TabPFN. Access our models through the cloud without requiring local GPU resources.

-   :material-application:{ .lg .middle } **User Interface**

    ---

    Visual interface for no-code interaction with TabPFN. Perfect for quick experimentation and visualization.

    [:octicons-arrow-right-24: Access GUI](https://ux.priorlabs.ai/)

-   :material-language-python:{ .lg .middle } **Python Package**

    ---

    Coming soon! Local installation with GPU support and scikit-learn compatible interface.

    [comment]: <> ([:octicons-arrow-right-24: Documentation](#))

-   :material-language-r:{ .lg .middle } **R Integration**

    ---

    Currently in development. Bringing TabPFN's capabilities to the R ecosystem for data scientists and researchers.

    [comment]: <> ([:octicons-arrow-right-24: Learn More](#))

</div>

## Installation

<!---
To install our software, we use pip the python package installer in combination with Git for code-management. An installation typically takes 5 minutes in a setup python environment. 
!!! tip
	
	The easiest way to install and run our code is via the Colab Notebooks shared in the link in our submission.
-->
You can access our models through our API (https://github.com/automl/tabpfn-client) or via our user interface built on top of the API (https://ux.priorlabs.ai/).
We will release open weights models soon, currently we are available via api and via our user interface built on top of the API.
=== "Python API Client (No GPU, Online)"

    ```bash
    pip install tabpfn-client
    ```

=== "Python Local (GPU)"

    !!! warning
        Not released yet


<!---
#### Software Dependencies and Operating Systems
Python: Version >= 3.9

Operating Systems: The software has been tested on major operating systems including:

- Ubuntu 20.04, 22.04

- Windows 10, 11

- macOS 11.0 (Big Sur) and later

Git Version 2 or later ([https://git-scm.com/](https://git-scm.com/))

#### Software Dependencies (as specified in `requirements.txt`):

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

#### Non-Standard Hardware
GPU: A CUDA-enabled GPU is recommended for optimal performance, though the software can also run on a CPU.


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
-->
<br>
<br>
