import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

#This function removes low variance features from the input dataframe.
def remLowVariance(X, threshold):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    return X[X.columns[selector.get_support(indices=True)]]

#This function selects the top k features based on the chi-squared test between each feature and the target variable.
def kBest(X, y, k):
    selector = SelectKBest(chi2, k=k)
    selector.fit(X, y)
    return X[X.columns[selector.get_support(indices=True)]]

#performs pca on dataset and returns transformed dataframe with n_components
def pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    column_names = [f"PC{i}" for i in range(1, n_components+1)]
    return pd.DataFrame(X_transformed, columns=column_names)

