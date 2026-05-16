import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Set up project structure
print("=" * 50)
print("DIABETES PREDICTION MODEL TRAINING")
print("=" * 50)

# Step 1: Load the dataset
print("\n[Step 1] Loading dataset...")
# Using the path we set up in your ml-project directory
data_path = '/home/ec2-user/ml-project/data/diabetes.csv'
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Step 2: Prepare features and target
print("\n[Step 2] Preparing features and target...")
X = df.iloc[:, :-1]  # All columns except the last one (Features)
y = df.iloc[:, -1]   # The last column (Target: 0 or 1)

print(f"Features shape: {X.shape}")
print(f"Target distribution:")
print(f"  - No diabetes (0): {(y == 0).sum()} patients")
print(f"  - Diabetes (1): {(y == 1).sum()} patients")

# Step 3: Split into training and test sets
print("\n[Step 3] Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Step 4: Train the Random Forest model
print("\n[Step 4] Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)
print("Training complete!")

# Step 5: Evaluate the model
print("\n[Step 5] Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Step 6: Save the trained model
print("\n[Step 6] Saving model...")
model_path = '/home/ec2-user/ml-project/models/diabetes_model.pkl'

# Ensure the models directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to: {model_path}")

# Step 7: Show feature importance
print("\n[Step 7] Feature Importance:")
feature_names = df.columns[:-1].tolist()
importances = model.feature_importances_
for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.4f}")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
