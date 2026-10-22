import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report
tags = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
t2i = {t: i for i, t in enumerate(tags)}
def rows(fname, url):
    docs, toks, tg = [], [], []
    for line in open(keras.utils.get_file(fname, url), encoding="utf-8"):
        p = line.split()
        if len(p) == 4 and p[0] != "-DOCSTART-":
            toks.append(p[0])
            tg.append(t2i[p[3]])
        elif toks:
            docs.append((toks, tg))
            toks, tg = [], []
    if toks:
        docs.append((toks, tg))
    return docs
tr = rows("eng.train", "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train")
te = rows("eng.testb", "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb")
vocab = {"<pad>": 0, "<unk>": 1}
for toks, _ in tr:
    for t in toks:
        vocab.setdefault(t, len(vocab))
def enc(rs):
    X = [([vocab.get(t, 1) for t in toks] + [0] * 60)[:60] for toks, _ in rs]
    Y = [(tg + [0] * 60)[:60] for _, tg in rs]
    return np.array(X), np.array(Y)
Xtr, Ytr = enc(tr)
m = keras.Sequential([keras.layers.Input((60,)), keras.layers.Embedding(len(vocab), 64, mask_zero=True), keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)), keras.layers.Dense(len(tags), activation="softmax")])
m.compile("adam", "sparse_categorical_crossentropy")
m.fit(Xtr, Ytr, epochs=3, batch_size=32, verbose=0)
pred = np.argmax(m.predict(enc(te[:500])[0], verbose=0), axis=-1)
yt, yp = [], []
for i, (toks, tg) in enumerate(te[:500]):
    n = min(len(toks), 60)
    yt += [tags[t] for t in tg[:n]]
    yp += [tags[t] for t in pred[i][:n]]
want = [t for t in tags if any(k in t for k in ("PER", "ORG", "LOC"))]
print(classification_report(yt, yp, labels=want))
# i = 0
# toks, tg = te[i]
# n = min(len(toks), 60)
# print("Sentence:", " ".join(toks))
# print("True:    ", [tags[t] for t in tg[:n]])
# print("Pred:    ", [tags[t] for t in pred[i][:n]])
