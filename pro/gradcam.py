"""
gradcam.py
----------
Grad-CAM for the fine-tuned MobileNetV2 model.
Works with TensorFlow 2.21 / Keras 3.x.

Root-cause of old errors
------------------------
Keras 3.x returns a PYTHON LIST when a multi-output model is called.
Doing  list[:, int]  fails with "list indices must be integers…".

Fix: split the model into two halves and watch the conv output
*before* computing class-scores through the head layers.
"""

import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from tensorflow.keras.preprocessing import image as keras_image


# ── 1. Build Grad-CAM helper objects ─────────────────────────────────────────
def build_grad_model(model, last_conv_layer_name: str = "out_relu"):
    """
    Returns (conv_model, head_layers) instead of a multi-output Model.

    conv_model  – sub-model that outputs the last conv layer activations
    head_layers – list of Keras layers that follow out_relu (GAP → Dense …)
    """
    # Sub-model: input → conv layer output
    conv_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = model.get_layer(last_conv_layer_name).output,
    )

    # Layers that come AFTER out_relu in the flat functional model
    names      = [l.name for l in model.layers]
    out_idx    = names.index(last_conv_layer_name)
    head_layers = model.layers[out_idx + 1:]          # e.g. GAP, Dense, Dropout, Dense

    return conv_model, head_layers


# ── 2. Core Grad-CAM for an image path (CLI / notebook) ──────────────────────
def get_gradcam_heatmap(img_path, model, grad_objects, class_names):
    """
    grad_objects : tuple returned by build_grad_model → (conv_model, head_layers)
    """
    conv_model, head_layers = grad_objects

    # Load & preprocess
    img       = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    img_tf    = tf.cast(img_array, tf.float32)

    # Get pred_index outside tape (clean Python int)
    preds      = model.predict(img_tf, verbose=0)
    pred_index = int(np.argmax(preds[0]))

    # Grad-CAM computation
    with tf.GradientTape() as tape:
        conv_out = conv_model(img_tf, training=False)
        tape.watch(conv_out)                    # watch BEFORE downstream ops
        x = conv_out
        for layer in head_layers:
            x = layer(x, training=False)
        class_score = x[:, pred_index]          # plain int index → no error

    grads        = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Colour overlay
    heatmap_u8      = np.uint8(255 * heatmap)
    heatmap_colored = cm.jet(heatmap_u8)[:, :, :3]
    heatmap_colored = tf.image.resize(heatmap_colored, (224, 224)).numpy()

    orig_img     = keras_image.img_to_array(
                       keras_image.load_img(img_path, target_size=(224, 224))
                   ) / 255.0
    superimposed  = np.clip(0.6 * orig_img + 0.4 * heatmap_colored, 0, 1)

    pred_class = class_names[pred_index]
    confidence = float(preds[0][pred_index]) * 100

    print(f"Prediction : {pred_class}")
    print(f"Confidence : {confidence:.2f}%")

    return orig_img, heatmap, superimposed, pred_class, confidence


# ── 3. Grad-CAM from a numpy array (Streamlit) ───────────────────────────────
def get_gradcam_from_array(img_array_norm, model, grad_objects, class_names):
    """
    img_array_norm : pre-processed numpy array, shape (1, 224, 224, 3), [0,1]
    grad_objects   : tuple returned by build_grad_model → (conv_model, head_layers)
    """
    conv_model, head_layers = grad_objects

    img_tf = tf.cast(img_array_norm, tf.float32)

    # Get pred_index outside the tape → guaranteed plain Python int
    preds      = model.predict(img_tf, verbose=0)
    pred_index = int(np.argmax(preds[0]))

    # Grad-CAM: watch conv output BEFORE head layers
    with tf.GradientTape() as tape:
        conv_out = conv_model(img_tf, training=False)
        tape.watch(conv_out)
        x = conv_out
        for layer in head_layers:
            x = layer(x, training=False)
        class_score = x[:, pred_index]          # plain int → works every time

    grads        = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_u8      = np.uint8(255 * heatmap)
    heatmap_colored = cm.jet(heatmap_u8)[:, :, :3]
    heatmap_colored = tf.image.resize(heatmap_colored, (224, 224)).numpy()

    orig_img     = img_array_norm[0]           # (224, 224, 3) already in [0,1]
    superimposed  = np.clip(0.6 * orig_img + 0.4 * heatmap_colored, 0, 1)

    pred_class = class_names[pred_index]
    confidence = float(preds[0][pred_index]) * 100

    return orig_img, heatmap, superimposed, pred_class, confidence


# ── Quick CLI test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model

    MODEL_PATH = r"..\model\CNN_MDOEL(FINE_TUNE).keras"
    IMAGE_PATH = r"..\Test_Images\sample-apple.jpg"

    CLASS_NAMES = [
        "freshapples", "freshbanana", "freshoranges",
        "rottenapples", "rottenbanana", "rottenoranges",
    ]

    model       = load_model(MODEL_PATH)
    grad_objects = build_grad_model(model)

    orig, hmap, overlay, pred, conf = get_gradcam_heatmap(
        IMAGE_PATH, model, grad_objects, CLASS_NAMES
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(orig);     axes[0].set_title("Original");          axes[0].axis("off")
    axes[1].imshow(hmap, cmap="jet"); axes[1].set_title("Heatmap");   axes[1].axis("off")
    axes[2].imshow(overlay);  axes[2].set_title(f"{pred} ({conf:.1f}%)"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("gradcam_result.png", dpi=150)
    plt.show()
