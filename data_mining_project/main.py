import numpy as np

from data_loader import load_wine_data
from preprocessing import scale_features
from pca_analysis import fit_pca, cumulative_variance, find_k_for_threshold
from visualisation import plot_cumulative_variance, plot_pca_scatter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def compare_classifier_accuracy(X_scaled, y, n_components):

    # Original features
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc_original = accuracy_score(y_test, pred)

    # PCA reduced features
    pca_k, X_pca_k = fit_pca(X_scaled, n_components=n_components)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_pca_k, y, test_size=0.25, random_state=42, stratify=y
    )
    clf2 = LogisticRegression(max_iter=500)
    clf2.fit(X_train2, y_train2)
    pred2 = clf2.predict(X_test2)
    acc_pca = accuracy_score(y_test2, pred2)

    return acc_original, acc_pca


def main():
    # 1) Load dataset
    df = load_wine_data("data/wine.data.csv")
    print("Dataset shape:", df.shape)
    print("Class distribution:\n", df["Class"].value_counts(), "\n")

    # 2) Split X and y
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 3) Standardize
    X_scaled = scale_features(X)

    # 4) Full PCA to compute variance
    pca_full, _ = fit_pca(X_scaled, n_components=None)
    cum_var = cumulative_variance(pca_full)

    # 5) Choose k for 95% variance
    threshold = 0.95
    k = find_k_for_threshold(cum_var, threshold=threshold)

    print("Explained variance ratio (first 5):", pca_full.explained_variance_ratio_[:5])
    print(f"Components needed for {threshold*100:.0f}% variance: {k}\n")

    # 6) PCA with 2 components for visualization
    pca2, X_pca2 = fit_pca(X_scaled, n_components=2)
    print("2D PCA variance ratio:", pca2.explained_variance_ratio_)
    print("Total variance explained by 2 components:", np.sum(pca2.explained_variance_ratio_), "\n")

    # 7) Visualizations
    plot_cumulative_variance(cum_var)
    plot_pca_scatter(X_pca2, y)

    # 8) Bonus: accuracy comparison
    acc_original, acc_pca = compare_classifier_accuracy(X_scaled, y, n_components=k)
    print("Logistic Regression accuracy (original features):", acc_original)
    print(f"Logistic Regression accuracy (after PCA with k={k}):", acc_pca)


if __name__ == "__main__":
    main()
