import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

train_ds, test_ds = tfds.load("cats_vs_dogs", split=["train[:80%]", "train[80%:]"], as_supervised=True)
def prep(x, y):
    return tf.image.resize(x, (160, 160)) / 255.0, tf.reshape(tf.cast(y, tf.float32), (1,))
train_ds = train_ds.map(prep).batch(32).take(30)
test_ds = test_ds.map(prep).batch(32).take(10)

def build(base):
    return keras.Sequential([base, keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(1, activation="sigmoid")])

base = keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights="imagenet")
base.trainable = False
fe = build(base)
fe.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
fe.fit(train_ds, epochs=2, verbose=0)
fe_acc = fe.evaluate(test_ds, verbose=0)[1]

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
ft = build(base)
ft.compile(optimizer=keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
ft.fit(train_ds, epochs=2, verbose=0)
ft_acc = ft.evaluate(test_ds, verbose=0)[1]

print("Feature extraction accuracy:", fe_acc)
print("Fine-tuning accuracy:", ft_acc)
