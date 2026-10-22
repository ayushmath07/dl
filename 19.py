import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib.patches as patches
det = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
def iou(a, b):
    x1, y1, x2, y2 = max(a[1], b[1]), max(a[0], b[0]), min(a[3], b[3]), min(a[2], b[2])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    return inter / ((a[3] - a[1]) * (a[2] - a[0]) + (b[3] - b[1]) * (b[2] - b[0]) - inter + 1e-9)
ious = []
for ex in tfds.load("voc/2007", split="test").take(5):
    img = ex["image"].numpy()
    out = det(ex["image"][None])
    boxes = out["detection_boxes"][0].numpy()[out["detection_scores"][0].numpy() > 0.4]
    for gt in ex["objects"]["bbox"].numpy():
        if len(boxes):
            ious.append(max(iou(gt, p) for p in boxes))
    fig, ax = plt.subplots()
    ax.imshow(img)
    for b in boxes:
        ax.add_patch(patches.Rectangle((b[1] * img.shape[1], b[0] * img.shape[0]), (b[3] - b[1]) * img.shape[1], (b[2] - b[0]) * img.shape[0], fill=False, edgecolor="red", linewidth=2))
    plt.axis("off")
    plt.show()
print("Mean IoU:", round(float(np.mean(ious)), 4))
# ex = next(iter(tfds.load("voc/2007", split="test")))
# img = ex["image"].numpy()
# out = det(ex["image"][None])
# boxes = out["detection_boxes"][0].numpy()[out["detection_scores"][0].numpy() > 0.4]
# fig, ax = plt.subplots()
# ax.imshow(img)
# [ax.add_patch(patches.Rectangle((b[1] * img.shape[1], b[0] * img.shape[0]), (b[3] - b[1]) * img.shape[1], (b[2] - b[0]) * img.shape[0], fill=False, edgecolor="red", linewidth=2)) for b in boxes]
# plt.axis("off"); plt.show()
