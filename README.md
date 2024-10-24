# MNIST Handwritten Digit Recognition
This project implements a machine learning model that recognizes handwritten digits from the MNIST dataset. The model is trained using a neural network and can predict digits from images provided by the user.

## Table of Contents
- Overview
- Dataset
- Model
- Prerequisites
- How to Run
- Features
- Results
- License
  
## Overview

The project aims to classify images of handwritten digits (0-9) from the MNIST dataset. It uses a neural network model trained on these images, which are 28x28 grayscale images of digits. The model can predict which digit is present in an input image provided by the user.

## Dataset
The MNIST dataset consists of 60,000 training images and 10,000 test images. Each image is 28x28 pixels and corresponds to one of the digits from 0 to 9.

## Model
The neural network architecture used in this project is a simple feedforward network that can be trained using backpropagation. The notebook uses TensorFlow and Keras to train and evaluate the model.

## Architecture
- Input layer: 28x28 grayscale images
- Dense layers with ReLU activation functions
- Softmax output layer to classify digits (0-9)

## Prerequisites
The following Python libraries are required:

- tensorflow
- numpy
- matplotlib
- opencv-python
Install the necessary packages using pip:
```
pip install tensorflow numpy matplotlib opencv-python
```

## How to Run
1. Clone the repository:

```
git clone https://github.com/your-username/mnist-digit-recognition.git
```
2. Navigate to the project directory:

```
cd mnist-digit-recognition
```

3. Run the Jupyter notebook: Open MNIST_PROJECT.ipynb in Jupyter and run all the cells to train the model and use the prediction functionality.

4. Predict your own image:

- Once the model is trained, you can input a custom image for prediction by specifying the path.
- The image will be resized to 28x28 pixels and converted to grayscale before making predictions.

## Features
- Model Training: 
  - Trains the neural network on the MNIST dataset.
- Image Prediction:
  - Predicts the digit from a custom image.
- Resizes and normalizes images before prediction.

- Visualization:
  - Displays the input image and predicted label using matplotlib.

## Results
After training, the model achieves high accuracy on the MNIST dataset and can effectively recognize handwritten digits.

## License
This project is licensed under the MIT License.
