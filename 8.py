import os, pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras

keras.utils.get_file("Flickr8k_Dataset.zip", "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip", extract=True)
keras.utils.get_file("Flickr8k_text.zip", "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip", extract=True)
home = pathlib.Path.home()
img_dir = str(next(home.glob(".keras/**/Flicker8k_Dataset", recursive=True)))
captions = {}
with open(next(home.glob(".keras/**/Flickr8k.token.txt", recursive=True))) as f:
    for line in f:
        img, cap = line.strip().split("\t")
        captions.setdefault(img.split("#")[0], []).append("startseq " + cap.strip().lower() + " endseq")
names = list(captions)[:100]

resnet = keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")
def extract(name):
    img = keras.utils.load_img(os.path.join(img_dir, name), target_size=(224, 224))
    x = keras.applications.resnet50.preprocess_input(keras.utils.img_to_array(img)[None])
    return resnet.predict(x, verbose=0)[0]
features = {n: extract(n) for n in names}

all_caps = [c for n in names for c in captions[n]]
wi = {w: i + 1 for i, w in enumerate(sorted({w for c in all_caps for w in c.split()}))}
iw = {i: w for w, i in wi.items()}
vocab = len(wi) + 1
max_len = max(len(c.split()) for c in all_caps)

X1, X2, Y = [], [], []
for n in names:
    for cap in captions[n]:
        seq = [wi[w] for w in cap.split()]
        for i in range(1, len(seq)):
            X1.append(features[n])
            X2.append(keras.utils.pad_sequences([seq[:i]], maxlen=max_len)[0])
            Y.append(seq[i])
X1, X2 = np.array(X1), np.array(X2)
Y = keras.utils.to_categorical(Y, vocab)

in1 = keras.Input(shape=(2048,))
fe = keras.layers.Dense(256, activation="relu")(in1)
in2 = keras.Input(shape=(max_len,))
se = keras.layers.LSTM(256)(keras.layers.Embedding(vocab, 256, mask_zero=True)(in2))
dec = keras.layers.Dense(256, activation="relu")(keras.layers.add([fe, se]))
out = keras.layers.Dense(vocab, activation="softmax")(dec)
model = keras.Model([in1, in2], out)
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit([X1, X2], Y, epochs=5, batch_size=64, verbose=0)

seq = [wi["startseq"]]
for _ in range(max_len):
    p = model.predict([features[names[0]][None], keras.utils.pad_sequences([seq], maxlen=max_len)], verbose=0)
    idx = int(np.argmax(p))
    if idx not in iw or iw[idx] == "endseq":
        break
    seq.append(idx)
print("Generated caption:", " ".join(iw[i] for i in seq[1:]))
