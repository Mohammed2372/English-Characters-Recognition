# English Character Recognition

This project focuses on the recognition of English characters using a Convolutional Neural Network (CNN) model. The model is trained on a dataset of images representing different English characters, and it is capable of accurately predicting the character in a given image. The project also includes a user-friendly GUI for selecting images and viewing predictions.

## Project Overview

### Features
- **CNN Model**: A Convolutional Neural Network is used to achieve high accuracy in character recognition.
- **Data Augmentation**: Techniques such as rotation, width shift, height shift, and zoom are used to enhance the training dataset.
- **Early Stopping and Model Checkpointing**: These techniques are used to prevent overfitting and save the best model during training.
- **GUI for Image Prediction**: A Tkinter-based GUI allows users to select an image, view the image, and get the predicted character label.

### Project Structure
- `english_character_recognition_model.keras`: Saved model file with the best weights.
- `label_encoder.pkl`: Serialized label encoder for decoding the predicted labels.
- `test.py`: Script for predicting the label of an image and running the GUI.
- `train.py`: Script for training the model.
- `README.md`: Project documentation.

### Kaggle Notebook
You can find the detailed analysis, model training process, and more in the Kaggle notebook:
[English Character Recognition - Kaggle](https://www.kaggle.com/code/mohammed237/english-characters-recognition)

## Installation and Usage

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.0 or higher
- scikit-learn
- numpy
- pandas
- Pillow
- Tkinter

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mohammed2372/English-Character-Recognition.git
   cd english-character-recognition
2. Install the required packages:
   ```bash
   pip install tensorflow scikit-learn numpy pandas pillow

### Training The Model
  run English Character Recognition.ipynb file

### Testing The Model
  ```bash
  python test.py
  ```

### Using the Model for Predictions
The test.py script allows you to select an image, view it, and see the predicted label in a graphical interface.


## Acknowledgements
- Kaggle for providing the dataset.
- TensorFlow and Keras for the deep learning framework.
- Scikit-learn for the label encoding and other utilities.
