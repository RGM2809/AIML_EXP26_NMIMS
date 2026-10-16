
# Step 1: Import the required libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 2: Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Step 3: Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

# Step 4: Train the model
model.fit(X, y)

# Step 5: Save the trained model
joblib.dump(model, "iris_model.joblib")

print("Model trained successfully!")
print("Model saved as iris_model.joblib")
print("Flower classes:", list(iris.target_names))
