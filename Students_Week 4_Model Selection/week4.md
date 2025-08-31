
---

# Lab 4 Report – Model Selection and Comparative Analysis

**Name:** Nandan D \
**Course:** Machine Learning\
**Lab:** Week 4

---

## Introduction

In this lab, I learned how to perform **hyperparameter tuning** and **model selection** for different classifiers. The lab was divided into two parts:

1. Writing my own **manual grid search** using loops and cross-validation.
2. Using scikit-learn’s **GridSearchCV** to perform the same task automatically.

The idea was to understand how grid search works internally and then compare it with the built-in efficient implementation.

---

## Datasets

I worked on the provided datasets. Each dataset was preprocessed (scaling, feature selection, encoding if needed) and then split into training and testing sets. Examples:

* **Wine Quality** → Predict good/bad quality wine.
* **HR Attrition** → Predict whether an employee will leave the company.
* **Banknote Authentication** → Predict if a banknote is genuine or forged.
* **QSAR Biodegradation** → Predict whether a chemical is biodegradable.

---

## Methodology

For every classifier, I used a **pipeline**:

`StandardScaler → SelectKBest → Classifier`

* **StandardScaler**: Normalized the data.
* **SelectKBest**: Picked the best features. The value of `k` was one of the parameters tuned.
* **Classifier**: I used three models: Decision Tree, k-Nearest Neighbors (kNN), and Logistic Regression.

### Part 1: Manual Grid Search

* I defined parameter grids for each classifier (e.g., max\_depth for Decision Tree, n\_neighbors for kNN, C for Logistic Regression).
* For each combination of parameters, I ran **5-fold Stratified Cross-Validation**.
* For every fold, I trained the pipeline, predicted probabilities, and calculated **ROC-AUC**.
* I stored the mean ROC-AUC for each combination and picked the best one.
* Finally, I retrained the best pipeline on the full training data.

### Part 2: GridSearchCV

* I built the same pipeline.
* I used **GridSearchCV** with the same parameter grids, 5-fold CV, and `scoring='roc_auc'`.
* GridSearchCV automatically tested all parameter combinations and gave me the best parameters, best score, and best estimator.

---

## Results and Observations

* Both manual grid search and GridSearchCV gave consistent results.
* GridSearchCV was much **faster and easier** to use.
* Manual implementation helped me understand exactly how hyperparameter tuning works internally.
* I generated confusion matrices and ROC curves for the final models.

---

## Conclusion

Through this lab, I understood the importance of:

* Using **pipelines** to avoid data leakage.
* Doing **hyperparameter tuning** to improve model performance.
* Using **cross-validation** for robust evaluation.
* The difference between **manual implementation** (good for learning) and **library implementation** (better for real-world use).

---
