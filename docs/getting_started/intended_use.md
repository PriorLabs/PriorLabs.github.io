# Usage tips

!!! note
    For a simple example getting started with classification see [classification tutorial](../tutorials/classification.md).

    We provide two comprehensive demo notebooks that guides through installation and functionalities. One [colab tutorial using the cloud](https://tinyurl.com/tabpfn-colab-online) and one [colab tutorial using the local GPU](https://tinyurl.com/tabpfn-colab-local).

### When to use TabPFN

TabPFN excels in handling small to medium-sized datasets with up to 10,000 samples and 500 features. For larger datasets, methods such as CatBoost, XGBoost, or AutoGluon are likely to outperform TabPFN.

### Intended Use of TabPFN

TabPFN is intended as a powerful drop-in replacement for traditional tabular data prediction tools, where top performance and fast training matter.
It still requires data scientists to prepare the data using their domain knowledge.
Data scientists will see benefits in performing feature engineering, data cleaning, and problem framing to get the most out of TabPFN.

### Limitations of TabPFN

1. TabPFN's inference speed may be slower than highly optimized approaches like CatBoost.
2. TabPFN's memory usage scales linearly with dataset size, which can be prohibitive for very large datasets.
3. Our evaluation focused on datasets with up to 10,000 samples and 500 features; scalability to larger datasets requires further study.

### Computational and Time Requirements

TabPFN is computationally efficient and can run inference on consumer hardware for most datasets. Training on a new dataset is recommended to run on a GPU as this speeds it up significantly. TabPFN is not optimized for real-time inference tasks, but V2 can perform much faster predictions than V1 of TabPFN.

### Data Preparation

TabPFN can handle raw data with minimal preprocessing. Provide the data in a tabular format, and TabPFN will automatically handle missing values, encode categorical variables, and normalize features. While TabPFN works well out-of-the-box, performance can further be improved using dataset-specific preprocessings.

### Interpreting Results

TabPFN's predictions come with uncertainty estimates, allowing you to assess the reliability of the results. You can use SHAP to interpret TabPFN's predictions and identify the most important features driving the model's decisions.

### Hyperparameter Tuning

TabPFN provides strong performance out-of-the-box without extensive hyperparameter tuning. If you have additional computational resources, you can automatically tune its hyperparameters using [post-hoc ensembling](https://github.com/PriorLabs/tabpfn-extensions/tree/main/src/tabpfn_extensions/post_hoc_ensembles) or [random tuning](https://github.com/PriorLabs/tabpfn-extensions/tree/main/src/tabpfn_extensions/hpo).
