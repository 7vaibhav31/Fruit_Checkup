"""
train_basic_cnn.py
------------------
Builds, trains, and evaluates the custom 4-block CNN from the notebook.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing import get_data_generators, BATCH_SIZE

# ── Get data ─────────────────────────────────────────────────────────────────
train_data, val_data = get_data_generators()

# ── Model ────────────────────────────────────────────────────────────────────
model = keras.Sequential([

    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 4
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Dense head
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation="softmax"),
])

model.summary()

# ── Compile ───────────────────────────────────────────────────────────────────
model.compile(
    optimizer = "adam",
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath       = "best_cnn_model.keras",
        monitor        = "val_accuracy",
        save_best_only = True,
        verbose        = 1,
    ),
    keras.callbacks.EarlyStopping(
        monitor              = "val_loss",
        patience             = 5,
        verbose              = 1,
        restore_best_weights = True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor = "val_loss",
        factor  = 0.5,
        patience= 3,
        verbose = 1,
    ),
]

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    train_data,
    steps_per_epoch  = train_data.samples  // BATCH_SIZE,
    epochs           = 15,
    validation_data  = val_data,
    validation_steps = val_data.samples    // BATCH_SIZE,
    callbacks        = callbacks,
    verbose          = 1,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(val_data)
print(f"\nValidation Accuracy: {acc:.4f}")
