from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
pip install imbalanced-learnpython -m pip install imbalanced-learn
from collections import Counter

# Load dataset
data = load_breast_cancer()


print("Keys:")
print(data.keys())

print("\nTarget Names:")
print(data.target_names)

print("\nNumber of Samples:", data.data.shape[0])
print("Number of Features:", data.data.shape[1])

print("\nFeature Names:")
print(data.feature_names)

print("\nShape of data.data:", data.data.shape)
print("Shape of data.target:", data.target.shape)



df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target



print("\nMissing Values Before:")
print(df.isnull().sum())

# Introduce 5 random missing values in mean radius
np.random.seed(42)
random_rows = np.random.choice(df.index, 5, replace=False)
df.loc[random_rows, "mean radius"] = np.nan

print("\nMissing Values After Introducing NaN:")
print(df["mean radius"].isnull().sum())

# Handle missing values


# Median is chosen because it is more robust to outliers
median_value = df["mean radius"].median()
df["mean radius"].fillna(median_value, inplace=True)

print("\nMissing Values After Filling:")
print(df["mean radius"].isnull().sum())

# 3. Check class distribution


print("\nClass Distribution:")
print(df["target"].value_counts())

print("\nTarget Labels:")
print("0 = Malignant")
print("1 = Benign")



X = df.drop("target", axis=1)
y = df["target"]

print("\nClass Counts Before SMOTE:")
print(Counter(y))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nClass Counts After SMOTE:")
print(Counter(y_resampled))