import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load your dataset from the .npz file
data = np.load("asl_landmarks.npz")
X = data["X"]  # shape: (num_samples, 63) or similar
y = data["y"]  # shape: (num_samples,)

# 2. Train a model (example: RandomForest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Save the trained model to a .pkl (or .joblib) file
joblib.dump(model, "model.pkl")