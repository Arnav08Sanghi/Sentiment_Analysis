import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_data_points (X, Y):

# Reduce to 2 dimensions for plotting
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    class_0 = X_reduced[Y == 0]
    class_1 = X_reduced[Y == 1]

    plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1')

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Data Visualization for Kernel Selection")
    plt.legend()
    plt.grid(True)
    plt.show()