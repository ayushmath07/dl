import tensorflow as tf
from tensorflow import keras

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[2.0, 0.0], [1.0, 3.0]])
print("Add:\n", a + b)
print("Multiply:\n", a * b)
print("Matmul:\n", tf.matmul(a, b))
print("Reshape:", tf.reshape(a, (4,)).numpy())
print("Sum:", tf.reduce_sum(a).numpy(), "Mean:", tf.reduce_mean(a).numpy())

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

perceptron = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(10, activation="softmax")])
perceptron.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
perceptron.fit(x_train, y_train, epochs=5, verbose=0)
p_acc = perceptron.evaluate(x_test, y_test, verbose=0)[1]

mlp = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(128, activation="relu"), keras.layers.Dense(10, activation="softmax")])
mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
mlp.fit(x_train, y_train, epochs=5, verbose=0)
m_acc = mlp.evaluate(x_test, y_test, verbose=0)[1]

print("Perceptron accuracy:", p_acc)
print("MLP accuracy:", m_acc)
