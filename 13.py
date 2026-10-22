import tensorflow as tf
from tensorflow.keras import layers
(xtr, ytr), (xte, yte) = tf.keras.datasets.fashion_mnist.load_data()
xtr, xte = xtr / 255.0, xte / 255.0
for units, act in [([128], 'relu'), ([256], 'relu'), ([128, 64], 'relu'), ([256, 128, 64], 'relu'), ([128, 64], 'sigmoid'), ([128, 64], 'tanh')]:
    ls = [layers.Flatten(input_shape=(28, 28))] + [layers.Dense(u, activation=act) for u in units] + [layers.Dense(10, activation='softmax')]
    m = tf.keras.Sequential(ls)
    m.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    m.fit(xtr, ytr, epochs=5, batch_size=128, verbose=0)
    print(units, act, round(m.evaluate(xte, yte, verbose=0)[1], 4))
