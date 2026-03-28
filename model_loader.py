"""
model_loader.py — Downloads and loads the 4 trained Keras models.

Models are hosted on Hugging Face Hub and cached locally.
Each model uses a different preprocessing function.
Supports lazy loading for Railway free-tier memory optimization.
"""

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
import keras

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION — Update these after uploading to HuggingFace
# ============================================================

# Your Hugging Face repo ID (format: "username/repo-name")
# Create a model repo at https://huggingface.co/new and upload the .keras files
HF_REPO_ID = os.environ.get("HF_REPO_ID", "muhammadhuzaifa2003/New_Model")


# Model filenames on Hugging Face Hub
MODEL_FILES = {
    "Sequential_CNN":  "Sequential_CNN_best.keras",
    "VGG16":           "VGG16_best.keras",
    "ResNet50":        "ResNet50_best.keras",
    "EfficientNetB0":  "EfficientNetB0_best.keras",
}

# Validation accuracies from training — used for weighted voting
VAL_ACCURACIES = {
    "Sequential_CNN":  0.9479,
    "VGG16":           0.9995,
    "ResNet50":        0.9947,
    "EfficientNetB0":  0.9879,
}

# Class names (must match the order from ImageDataGenerator)
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

# Grad-CAM target layers for each model
GRADCAM_LAYERS = {
    "VGG16":           "block5_conv3",
    "ResNet50":        "conv5_block3_out",
    "EfficientNetB0":  "top_conv",
}

IMG_SIZE = (224, 224)

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def preprocess_sequential(img_array):
    """Sequential CNN: simple rescale to [0, 1]."""
    return img_array / 255.0


def preprocess_vgg(img_array):
    """VGG16: uses caffe-style preprocessing (BGR, mean subtraction)."""
    return tf.keras.applications.vgg16.preprocess_input(img_array)


def preprocess_resnet(img_array):
    """ResNet50: uses caffe-style preprocessing."""
    return tf.keras.applications.resnet50.preprocess_input(img_array)


def preprocess_efficientnet(img_array):
    """EfficientNet: scale to [-1, 1]."""
    return tf.keras.applications.efficientnet.preprocess_input(img_array)


PREPROCESS_FUNCS = {
    "Sequential_CNN":  preprocess_sequential,
    "VGG16":           preprocess_vgg,
    "ResNet50":        preprocess_resnet,
    "EfficientNetB0":  preprocess_efficientnet,
}


# ============================================================
# MODEL LOADING
# ============================================================

class SafeDense(keras.layers.Dense):
    """
    Custom wrapper to swallow 'quantization_config' during deserialization 
    in older Keras versions (like 3.12.1 on Railway) that don't recognize it.
    """
    @classmethod
    def from_config(cls, config):
        if "quantization_config" in config:
            del config["quantization_config"]
        # In Keras 3, the config sometimes comes with nested 'config' or just flat.
        # We pass it up to the base Dense class which handles the rest.
        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        if "quantization_config" in config:
            del config["quantization_config"]
        return config


class ModelManager:
    """Manages downloading and loading of all 4 Keras models."""

    def __init__(self):
        self.models = {}
        self.loaded = False

    def download_model(self, model_name: str, max_retries: int = 3) -> str:
        """Download a model from Hugging Face Hub with retry logic. Returns local file path."""
        filename = MODEL_FILES[model_name]
        logger.info(f"Downloading {model_name} from HuggingFace ({HF_REPO_ID}/{filename})...")

        for attempt in range(1, max_retries + 1):
            try:
                local_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=filename,
                    cache_dir=os.environ.get("MODEL_CACHE_DIR", "/tmp/models"),
                )
                logger.info(f"  ✅ {model_name} downloaded to {local_path}")
                return local_path
            except Exception as e:
                logger.warning(f"  ⚠️ Attempt {attempt}/{max_retries} failed for {model_name}: {e}")
                if attempt < max_retries:
                    wait = 5 * attempt  # 5s, 10s, 15s
                    logger.info(f"  ⏳ Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Failed to download {model_name} after {max_retries} attempts: {e}")

    def load_all(self):
        """Download and load all 4 models into memory."""
        if self.loaded:
            logger.info("Models already loaded, skipping.")
            return

        logger.info("=" * 50)
        logger.info("Loading all models from Hugging Face Hub...")
        logger.info("=" * 50)

        for name in MODEL_FILES:
            try:
                local_path = self.download_model(name)
                # Pass SafeDense to hijack Dense layer deserialization and strip the unsupported kwarg
                self.models[name] = keras.models.load_model(
                    local_path, 
                    compile=False,
                    custom_objects={"Dense": SafeDense}
                )
                param_count = self.models[name].count_params()
                logger.info(f"  ✅ {name} loaded ({param_count:,} params)")
            except Exception as e:
                logger.error(f"  ❌ Failed to load {name}: {e}")

        self.loaded = True
        logger.info(f"\n✅ {len(self.models)}/{len(MODEL_FILES)} models loaded successfully!")

    def predict(self, img_array: np.ndarray) -> dict:
        """
        Run inference on all loaded models and return ensemble result.

        Args:
            img_array: Raw image as numpy array, shape (224, 224, 3), uint8 [0-255]

        Returns:
            dict with prediction, confidence, per-model results, etc.
        """
        if not self.models:
            raise RuntimeError("No models loaded! Call load_all() first.")

        model_results = []
        all_probabilities = []

        for name, model in self.models.items():
            # Prepare input: copy + preprocess
            preprocessed = PREPROCESS_FUNCS[name](np.copy(img_array).astype("float32"))
            input_batch = np.expand_dims(preprocessed, axis=0)  # (1, 224, 224, 3)

            # Predict
            probs = model.predict(input_batch, verbose=0)[0]  # (3,)
            pred_class_idx = int(np.argmax(probs))
            confidence = float(probs[pred_class_idx]) * 100

            model_results.append({
                "name": name if name != "Sequential_CNN" else "Custom CNN",
                "prediction": CLASS_NAMES[pred_class_idx],
                "confidence": round(confidence, 1),
            })
            all_probabilities.append(probs)

        # --- Weighted Soft Voting ---
        weights = np.array([VAL_ACCURACIES[n] for n in self.models.keys()])
        weights = weights / weights.sum()

        weighted_avg = np.average(all_probabilities, axis=0, weights=weights)
        ensemble_class_idx = int(np.argmax(weighted_avg))
        ensemble_confidence = float(weighted_avg[ensemble_class_idx]) * 100

        return {
            "prediction": CLASS_NAMES[ensemble_class_idx],
            "confidence_score": round(ensemble_confidence, 2),
            "models": model_results,
        }

    def predict_stream(self, img_array: np.ndarray):
        """
        Run inference yielding progress events on all loaded models.
        Yields dictionaries with step names and temporary states.
        """
        if not self.models:
            raise RuntimeError("No models loaded! Call load_all() first.")

        model_results = []
        all_probabilities = []

        yield {"event": "progress", "step": "Image Preprocessing"}

        for name, model in self.models.items():
            display_name = name if name != "Sequential_CNN" else "Custom CNN"
            yield {"event": "progress", "step": f"{display_name} Prediction"}
            
            # Prepare input: copy + preprocess
            preprocessed = PREPROCESS_FUNCS[name](np.copy(img_array).astype("float32"))
            input_batch = np.expand_dims(preprocessed, axis=0)  # (1, 224, 224, 3)

            # Predict
            probs = model.predict(input_batch, verbose=0)[0]  # (3,)
            pred_class_idx = int(np.argmax(probs))
            confidence = float(probs[pred_class_idx]) * 100

            model_results.append({
                "name": display_name,
                "prediction": CLASS_NAMES[pred_class_idx],
                "confidence": round(confidence, 1),
            })
            all_probabilities.append(probs)

            # Yield the result of THIS model so the frontend can populate it specifically!
            yield {"event": "model_result", "model": model_results[-1]}

        yield {"event": "progress", "step": "Ensemble Voting"}

        # --- Weighted Soft Voting ---
        weights = np.array([VAL_ACCURACIES[n] for n in self.models.keys()])
        weights = weights / weights.sum()

        weighted_avg = np.average(all_probabilities, axis=0, weights=weights)
        ensemble_class_idx = int(np.argmax(weighted_avg))
        ensemble_confidence = float(weighted_avg[ensemble_class_idx]) * 100

        yield {
            "event": "complete",
            "result": {
                "prediction": CLASS_NAMES[ensemble_class_idx],
                "confidence_score": round(ensemble_confidence, 2),
                "models": model_results,
            }
        }


# Global instance
model_manager = ModelManager()
