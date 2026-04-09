"""
train_mobilenetv2.py
--------------------
Transfer-learning with MobileNetV2, fine-tuned and saved as
CNN_MDOEL(FINE_TUNE).keras — matches the notebook exactly.
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras             import layers, models

from preprocessing import get_data_generators, BATCH_SIZE

# ── Data ─────────────────────────────────────────────────────────────────────
train_data, val_data = get_data_generators()

# ── Step 1 : Pretrained base (frozen) ────────────────────────────────────────
base_model           = MobileNetV2(
    input_shape  = (224, 224, 3),
    include_top  = False,       # remove ImageNet classifier
    weights      = "imagenet",
)
base_model.trainable = False    # freeze all base layers

print(f"Base model layers     : {len(base_model.layers)}")
print(f"Trainable params      : "
      f"{sum(w.numpy().size for w in base_model.trainable_weights)}")

# ── Step 2 : Custom classification head ──────────────────────────────────────
x      = base_model.output
x      = layers.GlobalAveragePooling2D()(x)
x      = layers.Dense(128, activation="relu")(x)
x      = layers.Dropout(0.5)(x)
output = layers.Dense(train_data.num_classes, activation="softmax")(x)

# ── Step 3 : Full model ───────────────────────────────────────────────────────
model = models.Model(
    inputs  = base_model.input,
    outputs = output,
)
model.summary()

# ── Step 4 : Compile & train (frozen base) ───────────────────────────────────
model.compile(
    optimizer = "adam",
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

model.fit(
    train_data,
    validation_data = val_data,
    epochs          = 10,
    verbose         = 1,
)

# ── Step 5 : Evaluate ─────────────────────────────────────────────────────────
loss, acc = model.evaluate(val_data)
print(f"\nValidation Accuracy (MobileNetV2 frozen): {acc:.4f}")

# ── Step 6 : Save the fine-tuned model ────────────────────────────────────────
SAVE_PATH = r"..\model\CNN_MDOEL(FINE_TUNE).keras"
model.save(SAVE_PATH)
print(f"\nModel saved → {SAVE_PATH}")
