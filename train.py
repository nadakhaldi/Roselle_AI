from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print("Recall (malignant):", recall_score(y_test, predictions, pos_label=0))

# Try different thresholds
proba_benign = model.predict_proba(X_test)[:, 1]

for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = (proba_benign >= t).astype(int)
    cm = confusion_matrix(y_test, pred)
    print("threshold", t,
          "| recall:", round(recall_score(y_test, pred, pos_label=0), 3),
          "| missed malignant:", cm[0][1],
          "| false alarms:", cm[1][0])