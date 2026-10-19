import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine , load_digits

wine = load_wine()

X = wine.data
y = wine.target
feat_names = wine.feature_names
target_names = wine.target_names

print ("Original Data shape ", X.shape)
print ("Original target shape ", y.shape)
print ("Features Names : ", feat_names)
print("Target Names : ", target_names)


scaler = StandardScaler()
X_scaled =  scaler.fit_transform(X)

pca = PCA(n_components=2)
X_reduces = pca.fit_transform(X_scaled)

print("original dataset shape : ", X.shape)
print("Reduced dataset Shape : ", X_reduces.shape)

wine_colors = ['Oragned', 'darked', 'crimson']
plt.figure(figsize=(10,6))

for class_num, color, target_names in zip(
    [0,1,2], wine_colors, target_names
):
    plt.scatter(
        X_reduces[y == class_num, 0],
        X_reduces[y == class_num, 1],
        color=color,
        label=target_names,
        alpha=0.7
    )

    plt.title('PCA of wine dataset')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

    print("Principal component loading :")
    print(pca.components_)

    print("Shape of principal component: ", pca.components_.shape)

    plt.figure(figsize=(14,5))

sns.heatmap(
    pca.components_,
    cmap='viridis',
    xticklabels=feat_names,
    yticklabels=['First Componet', 'Second Component']
)

plt.title('PCA Component Loadings')
plt.xlabel('Orignal Features')
plt.ylabel('Principal Components')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("Explained variance:")
print(pca.explained_variance_)

print(" Explained variance ratio")
print (pca.explained_variance_ratio_)

pca_3 = PCA(n_components=0.96)
X_reduces_96 = pca.fit_transform(X_scaled)

print("Numeber of componets selected: ", pca.n_components_)

print("Total explained Variance:",
      np.sum(pca_3.explained_variance_ratio_))

print("Reduced dataset shape:", X_reduces_96.shape)

