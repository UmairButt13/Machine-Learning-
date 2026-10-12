import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Wine dataset
wine = load_wine()

# Separate features (X) and target (y)
X = wine.data
y = wine.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Different k values
k_values = range(1, 30)

# Store accuracy scores
accuracy_scores = []

# Train and test KNN for each k value
for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Store accuracy
    accuracy_scores.append(accuracy)

# Find the best k value
best_k = k_values[accuracy_scores.index(max(accuracy_scores))]

# Find the highest accuracy
best_accuracy = max(accuracy_scores)

# Print results
print("Best value of k:", best_k)
print("Best accuracy:", best_accuracy)

# Plot Accuracy vs k
plt.scatter(k_values, accuracy_scores)

plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. k Values")

plt.show()