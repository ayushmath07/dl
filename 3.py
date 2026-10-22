import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation="relu", input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=0)
model.save("cnn_cifar10.keras")
loaded = keras.models.load_model("cnn_cifar10.keras")
loss, acc = loaded.evaluate(x_test, y_test, verbose=0)
print("Test loss:", loss, "Test accuracy:", acc)
# import matplotlib.pyplot as plt
# plt.imshow(x_test[0])
# plt.axis("off")
# plt.show()
# pred = loaded.predict(x_test[:1], verbose=0)
# print("Predicted class:", int(pred[0].argmax()), "Actual class:", int(y_test[0][0]))
