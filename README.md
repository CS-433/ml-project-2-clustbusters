# Road Segmentation from Satellite Images using CNN

## Project Overview
This project is centered on the segmentation of roads in satellite images using Convolutional Neural Networks (CNNs), particularly the U-Net architecture. The objective is to classify each pixel in the images as either road or background, leveraging deep learning techniques for accurate segmentation.

## Installation
Before starting, ensure the following libraries are installed:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- SciPy
- scikit-learn
- scikit-image

# Usage
## Data Preparation
The data preparation phase involves loading satellite images and their corresponding ground truth labels. Functions are provided for loading images, converting image formats, concatenating images with their labels, and cropping images into patches.

## Display Functions
Several functions are included for displaying images and overlays, such as overlaying predictions on the original images for visual comparison.

## Data Augmentation and Preprocessing
The project includes comprehensive data augmentation and preprocessing capabilities:
- Elastic deformation of images
- Random cropping and padding
- Resizing and padding images and masks
- Color augmentation
- Generating augmented datasets

## Model Building
A U-Net model is built for the task. The build_unet function allows for customizable parameters, including depth, dropout rates, and activation functions. The model's architecture is designed to effectively capture the spatial hierarchies in satellite images.

## Training and Validation
The project includes functionalities for training the U-Net model with the augmented dataset. It covers normalizing the training and test sets, converting ground truth to binary format, and defining class weights for balancing. The training process is monitored using callbacks like EarlyStopping and TensorBoard.

## Evaluation
Evaluate the model's performance on a validation set using metrics like the F1 score. Functions are provided for applying different thresholds to the model's predictions to optimize the F1 score.

## Prediction
The project includes functions for making predictions on test images, converting these predictions into segmented outputs, and displaying them.
