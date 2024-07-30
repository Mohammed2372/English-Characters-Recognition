import numpy as np
import os
import random
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import sys

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Parameters
img_size = (32, 32)
n = 20

# Load the trained model
model = tf.keras.models.load_model('english_character_recognition_model.keras')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=img_size, color_mode='grayscale')
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    return img_array

# Function to predict the label of the image
def predict_image(image_path):
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions, axis=1)
    return label_encoder.inverse_transform(predicted_label)[0]

# Function to select n random images from a folder
def select_random_images(folder_path, n):
    all_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return random.sample(all_images, n)

# Example usage
folder_path = 'Img'  # Replace with your image folder path
random_images = select_random_images(folder_path, n)

for image_path in random_images:
    predicted_label = predict_image(image_path)
    print(f'Image: {image_path} - Predicted Label: {predicted_label}')
