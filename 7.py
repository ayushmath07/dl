import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, y_train = x_train[:20000], y_train[:20000]

def build(init, opt):
    m = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation="relu", kernel_initializer=init, input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu", kernel_initializer=init),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu", kernel_initializer=init),
        keras.layers.Dense(10, activation="softmax")])
    m.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

schedule = keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=300, decay_rate=0.9)
experiments = [
    ("Xavier + Adam", "glorot_uniform", keras.optimizers.Adam(0.001)),
    ("He + Adam", "he_uniform", keras.optimizers.Adam(0.001)),
    ("He + SGD", "he_uniform", keras.optimizers.SGD(0.01)),
    ("He + SGD + LR schedule", "he_uniform", keras.optimizers.SGD(schedule)),
    ("He + RMSprop", "he_uniform", keras.optimizers.RMSprop(0.001)),
]
for name, init, opt in experiments:
    m = build(init, opt)
    h = m.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=0)
    acc = m.evaluate(x_test, y_test, verbose=0)[1]
    print(name, "| Val accuracy per epoch:", [round(a, 3) for a in h.history["val_accuracy"]], "| Test accuracy:", round(acc, 4))
# import matplotlib.pyplot as plt
# plt.imshow(x_test[0])
# plt.axis("off")
# plt.show()
# pred = m.predict(x_test[:1], verbose=0)
# print("Predicted class:", int(pred[0].argmax()), "Actual class:", int(y_test[0][0]))
