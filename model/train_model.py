"""
train_model.py
================
Trains the CIFAKE real-vs-AI-generated image classifier.

RUN THIS IN GOOGLE COLAB (free T4 GPU) -- not on your laptop, and not in
this repo's own runtime. Steps:

1. Open https://colab.research.google.com -> New Notebook.
2. Runtime -> Change runtime type -> T4 GPU.
3. Get a Kaggle API token: kaggle.com -> Account -> "Create New API Token"
   (downloads kaggle.json).
4. In the first Colab cell run:
       from google.colab import files
       files.upload()   # choose kaggle.json when prompted
       !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
       !pip install -q kagglehub
5. Paste this whole file into the next cell (or !wget it from your repo)
   and run it.
6. When it finishes, download cifake_cnn.h5 from the Colab file browser
   and drop it into app/saved_model/cifake_cnn.h5 in this project.

Training this tiny 2-conv-layer CNN on the 100k-image CIFAKE training
set takes roughly 5-10 minutes total on a T4.
"""

import os
import kagglehub
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------------------------------------------------------------
# 1. Download the CIFAKE dataset (120,000 images: 60k real CIFAR-10 +
#    60k Stable-Diffusion-generated equivalents), released publicly by
#    Bird & Lotfi (2024) on Kaggle.
# ---------------------------------------------------------------------
DATASET_PATH = kagglehub.dataset_download(
    "birdy654/cifake-real-and-ai-generated-synthetic-images"
)
print("Dataset downloaded to:", DATASET_PATH)

# The Kaggle release is organised as:
#   <root>/train/REAL, <root>/train/FAKE
#   <root>/test/REAL,  <root>/test/FAKE
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")

IMG_SIZE = (32, 32)
BATCH_SIZE = 64

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",   # alphabetical: FAKE=0, REAL=1 (matches the paper)
    shuffle=True,
    seed=1,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

print("Class order:", train_ds.class_names)  # should be ['FAKE', 'REAL']

# Normalise pixel values to [0, 1] and prefetch for speed.
normalization = layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------------------
# 2. Model -- the best-performing topology from the CIFAKE paper:
#    two Conv2D(32) + MaxPool blocks, one Dense(64) head, sigmoid output.
# ---------------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)

model.summary()

# ---------------------------------------------------------------------
# 3. Train, with early stopping so we don't overrun the Colab session.
# ---------------------------------------------------------------------
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    callbacks=[early_stop],
)

# ---------------------------------------------------------------------
# 4. Evaluate and save.
# ---------------------------------------------------------------------
results = model.evaluate(test_ds)
print(dict(zip(model.metrics_names, results)))

model.save("cifake_cnn.h5")
print("Saved model to cifake_cnn.h5 -- download this file from the Colab")
print("file browser and place it at app/saved_model/cifake_cnn.h5")
