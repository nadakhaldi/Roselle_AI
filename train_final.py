import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

THRESHOLD = 0.65
FEATURES = [
    "mean radius", "mean perimeter", "mean area",
    "mean compactness", "mean concavity", "mean concave points",
    "mean symmetry", "mean fractal dimension",
    "mean texture", "mean smoothness",
]

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X = df[FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
model.fit(X_train, y_train)

# Quick check with the final threshold
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= THRESHOLD).astype(int)
print(confusion_matrix(y_test, pred))

# Save the model and the settings the website will need
joblib.dump(model, "model.pkl")
info = {
    "features": FEATURES,
    "threshold": THRESHOLD,
    "ranges": {f: [float(X[f].min()), float(X[f].max())] for f in FEATURES},
}
with open("model_info.json", "w") as file:
    json.dump(info, file, indent=2)

print("Saved model.pkl and model_info.json")