# Usage tips

**When to use TabPFN**

TabPFN excels in handling small to medium-sized datasets with up to 10,000 samples and 500 features. For larger datasets, approaches such as CatBoost, XGB, or AutoGluon are likely to outperform TabPFN.

**Intended Use of TabPFN**

While TabPFN provides a powerful drop-in replacement for traditional tabular data models, achieving top performance on real-world problems often requires domain expertise and the ingenuity of data scientists. Data scientists should continue to apply their skills in feature engineering, data cleaning, and problem framing to get the most out of TabPFN.

**Limitations of TabPFN**

1. TabPFN's inference speed may be slower than highly optimized approaches like CatBoost.
2. TabPFN's memory usage scales linearly with dataset size, which can be prohibitive for very large datasets.
3. Our evaluation focused on datasets with up to 10,000 samples and 500 features; scalability to larger datasets requires further study.

**Computational and Time Requirements**

TabPFN is computationally efficient and can run on consumer hardware for most datasets. Training on a new dataset is recommended to run on a GPU as this speeds it up significantly. However, TabPFN is not optimized for real-time inference tasks.

**Data Preparation**

TabPFN can handle raw data with minimal preprocessing. Provide the data in a tabular format, and TabPFN will automatically handle missing values, encode categorical variables, and normalize features. While TabPFN works well out-of-the-box, performance can further be improved using dataset-specific preprocessings.

**Interpreting Results**

TabPFN's predictions come with uncertainty estimates, allowing you to assess the reliability of the results. You can use SHAP to interpret TabPFN's predictions and identify the most important features driving the model's decisions.

**Hyperparameter Tuning**

TabPFN provides strong performance out-of-the-box without extensive hyperparameter tuning. If you have additional computational resources, you can further optimize TabPFN's performance using random hyperparameter tuning or the Post-Hoc Ensembling (PHE) technique.