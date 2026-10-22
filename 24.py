import numpy as np
from tensorflow import keras
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
d = fetch_openml(data_id=42397)
X = StandardScaler().fit_transform(d.data.astype("float64"))
y = d.target.astype(int)
normal = X[y == 0]
tr, val = normal[:50000], normal[50000:60000]
te = np.vstack([normal[60000:80000], X[y == 1]])
yte = np.array([0] * 20000 + [1] * int((y == 1).sum()))
inp = keras.Input((X.shape[1],))
h = keras.layers.Dense(16, activation="relu")(inp)
h = keras.layers.Dense(8, activation="relu")(h)
h = keras.layers.Dense(16, activation="relu")(h)
ae = keras.Model(inp, keras.layers.Dense(X.shape[1])(h))
ae.compile("adam", "mse")
ae.fit(tr, tr, epochs=5, batch_size=256, verbose=0)
def err(a):
    return np.mean((ae.predict(a, verbose=0) - a) ** 2, axis=1)
thr = np.percentile(err(val), 99)
pred = (err(te) > thr).astype(int)
print("Threshold:", round(float(thr), 6))
print("Precision:", round(precision_score(yte, pred), 4), "Recall:", round(recall_score(yte, pred), 4), "F1:", round(f1_score(yte, pred), 4))
# idx = int(np.argmax((yte == 1) & (pred == 1)))
# e = err(te[idx:idx + 1])[0]
# print("Sample flagged transaction:", idx)
# print("Reconstruction error:", round(float(e), 6), "| threshold:", round(float(thr), 6), "| flagged:", bool(e > thr))
