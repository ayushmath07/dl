import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build(reg=None, dropout=False):
    m = keras.Sequential()
    m.add(keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],), kernel_regularizer=reg))
    m.add(keras.layers.Dense(32, activation="relu", kernel_regularizer=reg))
    if dropout:
        m.add(keras.layers.Dropout(0.3))
    m.add(keras.layers.Dense(1))
    m.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    return m

for name, reg, dr in [("Plain", None, False), ("L1", keras.regularizers.l1(0.01), False), ("L2", keras.regularizers.l2(0.01), False), ("Dropout", None, True)]:
    m = build(reg, dr)
    m.fit(X_train, y_train, epochs=50, verbose=0)
    print(name, "Train MSE:", round(m.evaluate(X_train, y_train, verbose=0), 4), "Test MSE:", round(m.evaluate(X_test, y_test, verbose=0), 4))
