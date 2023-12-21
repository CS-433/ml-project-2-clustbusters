
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import f1_score
from helper_functions import load_image


def predictions_to_majority_patches(predictions, patch_size=(16, 16)):
    """
    Convert model predictions to majority-voted patches.

    :param predictions: 2D array of predictions.
    :param patch_size: Tuple indicating the size of patches (height, width).
    :return: 2D array where each element represents the majority prediction for a patch.
    """
    height, width = predictions.shape[:2]
    patch_height, patch_width = patch_size
    num_patches_vertically = height // patch_height
    num_patches_horizontally = width // patch_width

    # Initialize an array to store the majority vote for each patch
    majority_patches = np.zeros((num_patches_vertically, num_patches_horizontally), dtype=np.uint8)

    # Iterate over the patches and calculate the majority value
    for i in range(num_patches_vertically):
        for j in range(num_patches_horizontally):
            patch = predictions[i * patch_height: (i + 1) * patch_height, j * patch_width: (j + 1) * patch_width]
            majority_value = np.argmax(np.bincount(patch.flatten()))
            majority_patches[i, j] = majority_value

    return majority_patches


def predictions_to_thresholded_patches(predictions, patch_size=(16, 16), threshold=0.5):
    """
    Convert model predictions to patches based on a threshold.

    :param predictions: 2D array of predictions.
    :param patch_size: Tuple indicating the size of patches (height, width).
    :param threshold: Proportion of foreground pixels required to label a patch as foreground.
    :return: 2D array where each element represents the label for a patch.
    """
    height, width = predictions.shape[:2]
    patch_height, patch_width = patch_size
    num_patches_vertically = height // patch_height
    num_patches_horizontally = width // patch_width

    # Initialize an array to store the label for each patch
    thresholded_patches = np.zeros((num_patches_vertically, num_patches_horizontally), dtype=np.uint8)

    # Iterate over the patches and calculate the label based on the threshold
    for i in range(num_patches_vertically):
        for j in range(num_patches_horizontally):
            patch = predictions[i * patch_height: (i + 1) * patch_height, j * patch_width: (j + 1) * patch_width]
            foreground_proportion = np.mean(patch)
            thresholded_patches[i, j] = int(foreground_proportion > threshold)

    return thresholded_patches


def evaluate_thresholds_batch(X_test, y_test, model, thresholds, batch_size=32):
    """
    Evaluate different thresholds for converting model predictions to binary masks using batch processing.

    :param X_test: Test set images.
    :param y_test: True masks for the test set.
    :param model: Trained model for making predictions.
    :param thresholds: List of thresholds to evaluate.
    :param batch_size: Batch size for processing predictions.
    :return: Dictionary with F1 scores for each threshold.
    """
    f1_scores = {threshold: 0 for threshold in thresholds}
    num_batches = len(X_test) // batch_size + (1 if len(X_test) % batch_size else 0)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_x = X_test[start_idx:end_idx]
        batch_y = y_test[start_idx:end_idx]

        # Predict for current batch
        batch_pred = model.predict(batch_x)
        batch_y_flat = batch_y.reshape(-1)
        
        for threshold in thresholds:
            # Apply threshold
            batch_pred_flat = (batch_pred > threshold).astype(np.uint8).reshape(-1)

            # Update F1 score calculation
            f1_scores[threshold] += f1_score(batch_y_flat, batch_pred_flat) * len(batch_x)

    # Average F1 scores over all batches
    f1_scores = {threshold: score / len(X_test) for threshold, score in f1_scores.items()}
    return f1_scores


def evaluate_thresholds(X_test, y_test, model, thresholds):
    """
    Evaluate different thresholds for converting model predictions to binary masks.

    :param X_test: Test set images.
    :param y_test: True masks for the test set.
    :param model: Trained model for making predictions.
    :param thresholds: List of thresholds to evaluate.
    :return: Dictionary with F1 scores for each threshold.
    """
    f1_scores = {}

    # Predict for all images at once
    predictions = model.predict(X_test)  # Assuming X_test is a numpy array
    predictions_flat = predictions.reshape(-1, predictions.shape[-1])  # Flatten predictions

    # Flatten all true masks
    y_test_flat = y_test.reshape(-1, y_test.shape[-1]).flatten()

    for threshold in thresholds:
        # Apply threshold to all predictions
        predicted_masks_flat = (predictions_flat > threshold).astype(np.uint8)

        # Compute F1 score
        f1 = f1_score(y_test_flat, predicted_masks_flat)
        f1_scores[threshold] = f1

    return f1_scores


def load_images_from_subfolders(root_dir, n=50):
    """
    Load images from subfolders within a specified directory.

    :param root_dir: Root directory containing image subfolders.
    :param n: Number of subfolders to process.
    :return: List of loaded images.
    """
    imgs = []

    for i in range(1, n + 1):
        subfolder_name = f"test_{i}"
        image_dir = os.path.join(root_dir, subfolder_name)
        
        if os.path.isdir(image_dir):
            files = os.listdir(image_dir)
            if files:
                image_path = os.path.join(image_dir, files[0])
                image = load_image(image_path)
                imgs.append(image)

    return np.array(imgs)


def predict_images(model, images, threshold=0.5):
    """
    Predict each image using the provided model.

    :param model: Trained model for making predictions.
    :param images: Numpy array of images to predict.
    :param threshold: Threshold to convert predictions to binary format.
    :return: Binary predictions for each image as a numpy array.
    """
    predictions = []

    for image in images:
        prediction = model.predict(image[tf.newaxis, ...])[0]
        predicted_mask = (prediction > threshold).astype(np.uint8)
        predictions.append(predicted_mask)

    return np.array(predictions)


def create_submission_entries(predictions, patch_size=16, threshold=0.25):
    """
    Create submission entries from predicted images.

    :param predictions: Numpy array of predicted images.
    :param patch_size: Size of the patches to be used for submission entries.
    :param threshold: Threshold for deciding the label of a patch.
    :return: DataFrame containing submission entries.
    """
    submission_entries = []

    for img_index in range(predictions.shape[0]):
        image = predictions[img_index, :, :, 0]  # Squeeze out the last dimension for single-channel images
        for j in range(0, image.shape[1], patch_size):  # Iterate columns
            for i in range(0, image.shape[0], patch_size):  # Iterate rows
                patch = image[i:i + patch_size, j:j + patch_size]
                label = 1 if np.mean(patch) > threshold else 0
                submission_entries.append([f"{img_index+1:03d}_{j}_{i}", label])

    return pd.DataFrame(submission_entries, columns=['id', 'prediction'])


def save_submission_to_csv(submission_df, filename):
    """
    Save the submission DataFrame to a CSV file.

    :param submission_df: DataFrame containing submission entries.
    :param filename: Name of the CSV file to save.
    """
    submission_df.to_csv(filename, index=False)
