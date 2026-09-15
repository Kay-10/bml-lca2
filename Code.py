# 1. Import all the necessary datasets
!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Load the Iris dataset
iris = fetch_ucirepo(id=53)
X = iris.data.features    # Features (sepal length, sepal width, petal length, petal width)
y = iris.data.target  # Target classes (Species: Setosa, Versicolor, Virginica)

# 3. Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)

# 4. Initialize the KNN Classifier (Choosing k = 3)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# 5. Train (fit) the model using the training data
knn.fit(X_train, y_train)

# 6. Make predictions on the test data
y_pred = knn.predict(X_test)

# 7. Determine and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of KNN Classifier (k={k}): {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
