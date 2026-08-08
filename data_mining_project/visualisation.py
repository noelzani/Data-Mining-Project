import matplotlib.pyplot as plt

def plot_cumulative_variance(cum_var):
    plt.figure()
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA - Cumulative Explained Variance (Wine Dataset)")
    plt.grid(True)
    plt.show()

def plot_pca_scatter(X_pca2, y):
    plt.figure()
    classes = sorted(y.unique())

    for cls in classes:
        idx = (y == cls)
        plt.scatter(X_pca2[idx, 0], X_pca2[idx, 1], label=f"Class {cls}")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Wine Dataset - PCA 2D Projection")
    plt.legend()
    plt.grid(True)
    plt.show()
