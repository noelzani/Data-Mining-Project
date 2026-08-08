from sklearn.decomposition import PCA
import numpy as np

def fit_pca(X_scaled, n_components=None):

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca

def cumulative_variance(pca):

    return np.cumsum(pca.explained_variance_ratio_)

def find_k_for_threshold(cum_var, threshold=0.95):

    return int(np.argmax(cum_var >= threshold) + 1)
