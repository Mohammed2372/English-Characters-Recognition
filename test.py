import os
import sys
import random
import joblib
import numpy as np
import tkinter as tk
import tensorflow as tf
from PIL import Image, ImageTk
from tkinter import filedialog, Label, Button
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Parameters
img_size = (32, 32)

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

# Tkinter GUI
def on_button_click():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        predicted_label = predict_image(file_path)
        result_label.config(text=f'Label is: {predicted_label}')
        
        # Display the selected image
        img = Image.open(file_path)
        img.thumbnail((200, 200))  # Resize the image to fit in the window
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img  # Keep a reference to avoid garbage collection

# Create the main window
root = tk.Tk()
root.title('Image Classification')
root.geometry('300x300')
# root.resizable(False, False)  # Make the window non-resizable

# Add a label above the button
instruction_label = Label(root, text="Choose an image")
instruction_label.pack(pady=10)

# Add a button to choose the image
choose_button = Button(root, text="Choose Image", command=on_button_click)
choose_button.pack(pady=10)

# Add a label to show the preview of the selected image
img_label = Label(root)
img_label.pack(pady=10)

# Add a label to show the prediction result
result_label = Label(root, text="")
result_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
