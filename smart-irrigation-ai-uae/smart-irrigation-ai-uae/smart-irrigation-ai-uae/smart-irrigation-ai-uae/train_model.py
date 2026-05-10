import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Encode text values
crop_encoder = LabelEncoder()
irrigation_encoder = LabelEncoder()

data['crop_type'] = crop_encoder.fit_transform(data['crop_type'])
data['irrigation_needed'] = irrigation_encoder.fit_transform(data['irrigation_needed'])

# Features and target
X = data[['temperature', 'humidity', 'soil_moisture', 'rainfall', 'crop_type']]
y = data['irrigation_needed']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

# Save model
joblib.dump(model, "irrigation_model.pkl")

print("Model trained successfully!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")