"""
gradcam.py — Generates Grad-CAM heatmap visualizations.

Produces a heatmap overlay showing which regions of the CT scan
influenced the model's prediction most strongly.
"""

import cv2
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image

_grad_models_cache = {}
_grad_funcs_cache = {}


def get_grad_step_func(grad_model):
    """Wrap the gradient computation in a compiled static graph for maximum performance."""
    @tf.function
    def compute_gradients(img_array):
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv output by the pooled gradients
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize: ReLU + scale to [0, 1]
        # Must use tf.where (not Python if) inside @tf.function graph mode,
        # otherwise the condition is only evaluated once at trace time and can
        # produce a permanently all-zero (black) heatmap.
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        heatmap = tf.where(max_val > 0, heatmap / max_val, heatmap)

        return heatmap

    return compute_gradients


def generate_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """
    Generate a Grad-CAM heatmap for a given image and model.

    Args:
        model: Loaded Keras model
        img_array: Preprocessed image, shape (1, 224, 224, 3)
        last_conv_layer_name: Name of the last convolutional layer

    Returns:
        heatmap: numpy array of shape (224, 224), values in [0, 1]
    """
    cache_key = id(model)

    # 1. Cache the sub-model slicing so we don't rebuild memory-heavy graphs every upload
    if cache_key not in _grad_models_cache:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )
        _grad_models_cache[cache_key] = grad_model
        _grad_funcs_cache[cache_key] = get_grad_step_func(grad_model)

    # 2. Run the compiled graph
    heatmap_tensor = _grad_funcs_cache[cache_key](img_array)

    # Ensure float32 numpy array
    result = heatmap_tensor.numpy()
    return np.asarray(result, dtype=np.float32)


def create_gradcam_overlay(original_img, heatmap, alpha=0.4):
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_img: BGR image, shape (224, 224, 3), uint8
        heatmap: heatmap array, shape (H, W), values in [0, 1]
        alpha: opacity of the heatmap overlay

    Returns:
        overlay: BGR image with heatmap overlay, uint8
    """
    # Resize heatmap to match image dimensions
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Apply JET colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    # Blend with original image
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def gradcam_to_base64(original_img_rgb, model, preprocessed_img, last_conv_layer_name):
    """
    Full Grad-CAM pipeline: generate heatmap → overlay → encode as base64 JPEG.

    Args:
        original_img_rgb: RGB image, shape (224, 224, 3), uint8
        model: Loaded Keras model
        preprocessed_img: Preprocessed image batch, shape (1, 224, 224, 3)
        last_conv_layer_name: Name of the target conv layer

    Returns:
        base64 string of the overlay JPEG image (with data URI prefix), or None on failure
    """
    try:
        # Generate heatmap
        heatmap = generate_gradcam_heatmap(model, preprocessed_img, last_conv_layer_name)

        # Convert RGB to BGR for OpenCV overlay
        original_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)

        # Create overlay
        overlay_bgr = create_gradcam_overlay(original_bgr, heatmap, alpha=0.4)

        # Convert back to RGB for PIL
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # Encode as base64 JPEG (much faster and smaller than PNG)
        pil_img = Image.fromarray(overlay_rgb)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        base64_str = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"

    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return None
