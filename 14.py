import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
for act in ['relu', 'sigmoid', 'tanh']:
    for opt in ['adam', 'sgd', 'rmsprop']:
        m = tf.keras.Sequential([layers.Dense(32, activation=act, input_shape=(30,)), layers.Dense(16, activation=act), layers.Dense(1, activation='sigmoid')])
        m.compile(opt, 'binary_crossentropy', metrics=['accuracy'])
        m.fit(xtr, ytr, epochs=30, batch_size=16, verbose=0)
        print(act, opt, round(m.evaluate(xte, yte, verbose=0)[1], 4))
