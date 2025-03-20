import os
import cv2
import numpy as np
import tensorflow as tf

# Path to the trained U-Net model (update if your file name is different).
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_unet_model.h5')

# Load the U-Net model once at import time
model = tf.keras.models.load_model(MODEL_PATH)
print("U-Net model loaded from:", MODEL_PATH)

IMG_HEIGHT, IMG_WIDTH = 224, 224

def load_and_preprocess_image(image_path):
    """
    Loads an image, resizes it to (224, 224), normalizes to [0,1].
    Returns the preprocessed image or None if there's an error.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype('float32') / 255.0
    # Expand dimensions to have shape (1, 224, 224, 3)
    return np.expand_dims(image, axis=0)

def predict_temperature_map(image_path):
    """
    Runs the loaded U-Net model on the given image to predict the temperature map.
    Returns a 2D numpy array of shape (224,224) with predicted temperatures.
    """
    img = load_and_preprocess_image(image_path)
    if img is None:
        return None
    # Model predicts shape (1, 224, 224, 1)
    pred_map = model.predict(img)[0, :, :, 0]
    return pred_map