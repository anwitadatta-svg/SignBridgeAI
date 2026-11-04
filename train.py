import csv

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

with open("data.csv") as f:
    rows = list(csv.reader(f))[1:]

X = np.array([[float(v) for v in r[1:]] for r in rows])
y = np.array([r[0] for r in rows])
print("Samples:", len(y), "Letters:", sorted(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {acc:.1%}")

joblib.dump(model, "model.pkl")
print("Saved model.pkl")