import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

chosen = [
    "mean radius", "mean perimeter",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean texture", "mean smoothness",
]

def run(columns, label):
    X_train, X_test, y_train, y_test = train_test_split(
        df[columns], y, test_size=0.2, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.9).astype(int)
    cm = confusion_matrix(y_test, pred)
    print(label,
          "| accuracy:", round(accuracy_score(y_test, pred), 3),
          "| recall:", round(recall_score(y_test, pred, pos_label=0), 3),
          "| missed malignant:", cm[0][1],
          "| false alarms:", cm[1][0])

run(list(data.feature_names), "all 30 features")
run(chosen, "8 chosen features")

chosen10 = chosen + ["mean area", "mean fractal dimension"]

def scan(columns, label):
    X_train, X_test, y_train, y_test = train_test_split(
        df[columns], y, test_size=0.2, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    for t in [0.5, 0.7, 0.8, 0.9]:
        pred = (proba >= t).astype(int)
        cm = confusion_matrix(y_test, pred)
        print(label, "| threshold", t,
              "| missed malignant:", cm[0][1],
              "| false alarms:", cm[1][0])

scan(chosen, "8 features")
scan(chosen10, "10 features")