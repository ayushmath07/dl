import zipfile
import numpy as np
from tensorflow import keras
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
z = keras.utils.get_file("fra-eng.zip", "http://www.manythings.org/anki/fra-eng.zip")
with zipfile.ZipFile(z) as f:
    lines = f.read("fra.txt").decode("utf-8").strip().split("\n")
pairs = []
for l in lines:
    en, fr = l.split("\t")[:2]
    en, fr = en.lower().strip(), "<start> " + fr.lower().strip() + " <end>"
    if len(en.split()) <= 10 and len(fr.split()) <= 12:
        pairs.append((en, fr))
pairs = pairs[:10000]
def vocab(texts):
    d = {"<pad>": 0, "<unk>": 1}
    for w in sorted(set(" ".join(texts).split())):
        d[w] = len(d)
    return d
ev, fv = vocab([p[0] for p in pairs]), vocab([p[1] for p in pairs])
i2f = {i: w for w, i in fv.items()}
def encode(texts, d):
    return keras.utils.pad_sequences([[d.get(w, 1) for w in t.split()] for t in texts], padding="post")
X, Y = encode([p[0] for p in pairs], ev), encode([p[1] for p in pairs], fv)
enc_in = keras.Input(shape=(None,))
enc_out, sh, sc = keras.layers.LSTM(256, return_sequences=True, return_state=True)(keras.layers.Embedding(len(ev), 256, mask_zero=True)(enc_in))
dec_in = keras.Input(shape=(None,))
dec_out, _, _ = keras.layers.LSTM(256, return_sequences=True, return_state=True)(keras.layers.Embedding(len(fv), 256, mask_zero=True)(dec_in), initial_state=[sh, sc])
att = keras.layers.Attention()([dec_out, enc_out])
out = keras.layers.Dense(len(fv), activation="softmax")(keras.layers.Concatenate()([dec_out, att]))
m = keras.Model([enc_in, dec_in], out)
m.compile("adam", "sparse_categorical_crossentropy")
m.fit([X, Y[:, :-1]], Y[:, 1:], epochs=6, batch_size=64, verbose=0)
def translate(src):
    es = encode([src], ev)
    ds = [fv["<start>"]]
    for _ in range(15):
        k = int(np.argmax(m.predict([es, np.array([ds])], verbose=0)[0, -1]))
        if k == fv["<end>"]:
            break
        ds.append(k)
    return [i2f[k] for k in ds[1:]]
refs = [[p[1].replace("<start> ", "").replace(" <end>", "").split()] for p in pairs[:100]]
hyps = [translate(p[0]) for p in pairs[:100]]
print("BLEU:", round(corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1), 4))
for p in pairs[:3]:
    print(p[0], "->", " ".join(translate(p[0])))
# src = pairs[100][0]
# print("EN:", src)
# print("FR (true):", pairs[100][1].replace("<start> ", "").replace(" <end>", ""))
# print("FR (pred):", " ".join(translate(src)))
