import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# --- Calibration Parameters ---
# These values map pixel intensity (0-255) to temperature values.
min_temp = -40  # Temperature corresponding to pixel value 0
max_temp = 80   # Temperature corresponding to pixel value 255

def calibration_function(pixel_values):
    """
    Linearly map pixel intensity (0-255) to temperature.
    """
    temperature = (pixel_values / 255.0) * (max_temp - min_temp) + min_temp
    return temperature

# --- Preprocessing Function ---
img_height, img_width = 224, 224

def load_and_preprocess_image(image_path):
    """
    Loads an image, resizes it, and computes a temperature map from its grayscale version.
    Returns:
      - image_float: the normalized color image (for input to the network)
      - temp_map: the computed temperature map (shape: H x W x 1) as the target.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    # Resize image to desired size
    image = cv2.resize(image, (img_width, img_height))
    
    # Normalize image for network input (0-1 range)
    image_float = image.astype(np.float32) / 255.0
    
    # Convert image to grayscale for calibration target.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temp_map = calibration_function(gray)
    # Expand dims to have shape (H, W, 1)
    temp_map = np.expand_dims(temp_map, axis=-1)
    
    return image_float, temp_map

# --- Gather Image Paths Recursively ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "Mahasat_therm-2")

# Recursively search for image files in the specified directory and its subdirectories.
image_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
            image_files.append(os.path.join(root, file))

if not image_files:
    raise ValueError("No image files found in the specified directory.")

# Split the image file paths into training and validation sets.
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# --- Data Generator ---
def data_generator(file_list):
    for image_path in file_list:
        img, target = load_and_preprocess_image(image_path)
        if img is not None and target is not None:
            yield img, target

batch_size = 8

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_files),
    output_types=(tf.float32, tf.float32),
    output_shapes=((img_height, img_width, 3), (img_height, img_width, 1))
)
train_dataset = train_dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_files),
    output_types=(tf.float32, tf.float32),
    output_shapes=((img_height, img_width, 3), (img_height, img_width, 1))
)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Build a Simple U-Net Model for Temperature Map Regression ---
def build_unet(input_size=(img_height, img_width, 3)):
    inputs = Input(input_size)
    
    # Downsampling
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    
    # Upsampling
    u2 = UpSampling2D((2, 2))(c3)
    concat2 = Concatenate()([u2, c2])
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat2)
    
    u1 = UpSampling2D((2, 2))(c4)
    concat1 = Concatenate()([u1, c1])
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(concat1)
    
    # Final layer outputs a single channel temperature map.
    outputs = Conv2D(1, (1, 1), activation='linear')(c5)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_unet()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
model.summary()

# --- Callbacks ---
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_unet_model.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# --- Training ---
epochs = 20  # Adjust as needed
history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=val_dataset,
                    callbacks=[early_stop, checkpoint, reduce_lr])

val_loss, val_mae = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

# --- Testing / Visualization on a Sample Image ---
# Pick a test image from the validation set.
test_image_path = val_files[0]
img, true_temp_map = load_and_preprocess_image(test_image_path)

# Predict the temperature map using the trained model.
pred_temp_map = model.predict(np.expand_dims(img, axis=0))[0, :, :, 0]  # Shape: (224,224)

# Create coordinate grid for contour plotting.
height, width = pred_temp_map.shape
x = np.arange(0, width)
y = np.arange(0, height)
X, Y = np.meshgrid(x, y)

# Plot the input image and overlay predicted temperature contours.
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title("Input Image")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
contours = plt.contour(X, Y, pred_temp_map, levels=15, cmap='jet', alpha=0.7)
plt.colorbar(contours, label='Temperature (°C)')
plt.title("Predicted Temperature Contours")
plt.show()

# --- Interactive Click Event ---s
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x_click = int(event.xdata)
        y_click = int(event.ydata)
        temp_val = pred_temp_map[y_click, x_click]
        print(f"Temperature at (x={x_click}, y={y_click}): {temp_val:.2f} °C")

# Connect the click event to the plot.
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()