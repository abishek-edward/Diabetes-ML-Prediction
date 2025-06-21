import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load Data
df = pd.read_csv("diabetes.csv")
print("Dataset Loaded Successfully!\n")
print(df.head())

# Step 2: Check for Nulls
print("\nNull Values:\n", df.isnull().sum())

# Step 3: Data Visualization
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 4: Split Data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 6: Evaluate Model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save Model
joblib.dump(model, "diabetes_model.pkl")
print("\nModel saved as 'diabetes_model.pkl'")
