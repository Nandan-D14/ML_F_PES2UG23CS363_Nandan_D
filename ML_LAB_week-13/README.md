# Bank Customer Segmentation Analysis

This project performs a comprehensive analysis and customer segmentation on the Bank Marketing dataset. The primary goal is to partition customers into distinct groups using K-means clustering and various other data analysis techniques. All the analysis and implementation are detailed in the `PES2UG23CS363_Lab_Week13.ipynb` Jupyter Notebook.

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
  - [2. Dimensionality Reduction with PCA](#2-dimensionality-reduction-with-pca)
  - [3. K-Means Clustering](#3-k-means-clustering)
  - [4. Advanced Clustering Techniques (Bonus)](#4-advanced-clustering-techniques-bonus)
- [Results and Visualizations](#results-and-visualizations)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Objective

The main objective of this lab is to apply unsupervised machine learning techniques to segment bank customers based on their demographic and transactional data. This can help the bank to better understand its customer base and tailor its marketing strategies.

## Dataset

The dataset used is `bank-full.csv`, which contains 45,211 records of bank clients. The features include:

- **Demographic Information**: age, job, marital status, education.
- **Financial Information**: balance, default, housing loan, personal loan.
- **Campaign Information**: campaign (number of contacts performed during this campaign), previous (number of contacts performed before this campaign).

## Methodology

### 1. Data Loading and Preprocessing

The initial step involves preparing the data for the clustering algorithms. This includes:

- **Loading the Data**: The `bank-full.csv` dataset is loaded into a pandas DataFrame.
- **Categorical Feature Encoding**: Categorical features like `job`, `marital`, `education`, etc., are converted into numerical format using `sklearn.preprocessing.LabelEncoder`.
- **Numerical Feature Scaling**: Numerical features such as `age`, `balance`, `campaign`, and `previous` are scaled using `sklearn.preprocessing.StandardScaler`. This ensures that all features contribute equally to the distance calculations in the clustering algorithm.
- **Correlation Analysis**: A heatmap is generated to visualize the correlation between the features.

### 2. Dimensionality Reduction with PCA

To visualize the clusters in a 2D space, Principal Component Analysis (PCA) is applied to reduce the feature space to two principal components. The cumulative explained variance of these two components is approximately 28.12%.

### 3. K-Means Clustering

A from-scratch implementation of the K-means algorithm is provided in the `KMeansClustering` class. The process is as follows:

- **Initialization**: Centroids are initialized by randomly selecting data points from the dataset.
- **Assignment Step**: Each data point is assigned to the nearest centroid based on Euclidean distance.
- **Update Step**: The centroids are recalculated as the mean of all data points assigned to each cluster.
- **Convergence**: The algorithm iterates until the centroids no longer change significantly or the maximum number of iterations is reached.

To determine the optimal number of clusters (`k`), two methods are used:
- **Elbow Method**: This method plots the inertia (within-cluster sum of squares) for different values of `k`. The "elbow" of the curve suggests the optimal `k`.
- **Silhouette Score**: The silhouette score measures how similar a data point is to its own cluster compared to other clusters. The `k` with the highest silhouette score is chosen. In this analysis, the optimal number of clusters was found to be **3**.

### 4. Advanced Clustering Techniques (Bonus)

Several advanced techniques were implemented to enhance the clustering analysis:

- **K-means++ Initialization**: This technique provides a smarter way to initialize the centroids, which often leads to better and more consistent results compared to random initialization.
- **Bisecting K-means**: A hierarchical clustering algorithm that recursively splits the largest cluster into two until the desired number of clusters is reached.
- **Manhattan Distance**: The K-means algorithm was modified to use the Manhattan (L1) distance metric instead of the default Euclidean (L2) distance. The silhouette score was used to compare the performance.
- **Cluster Interpretation**: The characteristics of each of the three clusters were analyzed by examining the mean values of numerical features and the mode of categorical features.
- **Outlier Detection**: Outliers were identified as data points that are far from their assigned cluster's centroid (beyond 3 standard deviations from the mean distance).

## Results and Visualizations

The notebook generates several visualizations to illustrate the results of the analysis:

1.  **Feature Correlation Heatmap**: Shows the correlation between different features.
    ![Feature Correlation Heatmap](feature_correlation_heatmap.png)

2.  **PCA Transformation Plots**: Visualizes the explained variance by the principal components and the data distribution in the 2D PCA space.
    ![PCA Transformation Plots](pca_transformation_plots.png)

3.  **Elbow Method and Silhouette Score Plots**: Used to determine the optimal number of clusters.
    ![Elbow Method Plot](elbow_method_plot.png)
    ![Silhouette Analysis Plot](silhouette_analysis_plot.png)

4.  **Final Clustering Visualization**: Shows the final three clusters in the 2D PCA space, along with the distribution of data points in each cluster.
    ![Final Clustering Visualization](final_clustering_visualization.png)

5.  **Cluster Characteristics Plots**: Box plots showing the distribution of `age` and `balance` for each cluster.
    ![Cluster Characteristics Plots](cluster_characteristics_plots.png)

6.  **Clustering with Outlier Visualization**: Highlights the identified outliers in the clustered data.
    ![Clustering Outlier Visualization](clustering_outlier_visualization.png)

## How to Run

1.  Clone this repository to your local machine.
2.  Ensure you have Python and Jupyter Notebook installed.
3.  Install the required dependencies listed below.
4.  Open and run the `PES2UG23CS363_Lab_Week13.ipynb` notebook in a Jupyter environment.

## Dependencies

The following Python libraries are required to run the notebook:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install them using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```