
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image


def load_image(infilename):
    """
    Load an image from a file.
    
    :param infilename: Path to the image file.
    :return: Image data as a numpy array.
    """
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    """
    Convert an image from floating point to uint8 format.
    
    :param img: Floating point image data.
    :return: Image data in uint8 format.
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    """
    Concatenate an image with its ground truth (label) image.
    
    :param img: Original image as a numpy array.
    :param gt_img: Ground truth image (label image) as a numpy array.
    :return: Concatenated image.
    """
    gt_img = np.squeeze(gt_img)
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    """
    Crop an image into patches of a specified size.
    
    :param im: Image data as a numpy array.
    :param w: Width of each patch.
    :param h: Height of each patch.
    :return: List of image patches.
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches


def load_images_from_directory(directory, file_names, max_images=100):
    """
    Load a specified number of images from a directory.

    :param directory: Path to the directory containing images.
    :param file_names: List of file names to load.
    :param max_images: Maximum number of images to load.
    :return: List of loaded images.
    """
    images = [load_image(os.path.join(directory, file_name)) for file_name in file_names[:max_images]]
    return images


def load_dataset(root_dir, loader_function, max_images=None):
    """
    Loads a dataset of images and their corresponding ground truth images.

    Args:
    root_dir (str): The root directory containing the 'images' and 'groundtruth' subdirectories.
    loader_function (function): Function to load images from a directory. 
                                This function should take directory path, list of filenames, 
                                and max_images as arguments.
    max_images (int, optional): Maximum number of images to load. Default is None (load all images).

    Returns:
    tuple: A tuple containing two numpy arrays, one for original images and one for ground truth images.
    """
    # Load original images
    image_dir = os.path.join(root_dir, "images/")
    image_files = os.listdir(image_dir)
    n = min(len(image_files), max_images) if max_images is not None else len(image_files)
    imgs = loader_function(image_dir, image_files, max_images=n)
    print(f"Loading {len(imgs)} original images")

    # Load ground truth images
    gt_dir = os.path.join(root_dir, "groundtruth/")
    gt_imgs = loader_function(gt_dir, image_files, max_images=n)
    print(f"Loading {len(gt_imgs)} ground truth images")
    gt_imgs = np.expand_dims(gt_imgs, axis=-1)  # Add an extra dimension

    return imgs, gt_imgs


def make_img_overlay(img, predicted_img):
    """
    Create an overlay image showing the prediction on top of the original image.

    :param img: Original image as a numpy array.
    :param predicted_img: Prediction image as a numpy array (binary mask).
    :return: Image object with overlay.
    """
    # Squeeze the singleton dimension from predicted_img if it exists
    predicted_img = np.squeeze(predicted_img, axis=-1)

    # Get dimensions of the original image
    w, h = img.shape[0], img.shape[1]

    # Create an empty color mask with the same dimensions
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)

    # Set the red channel of the mask to the prediction mask * 255
    color_mask[:, :, 0] = predicted_img * 255

    # Convert the original image to an 8-bit format (0-255 range)
    img8 = img_float_to_uint8(img)

    # Convert numpy arrays to PIL Image objects for blending
    # Convert to RGBA for alpha channel (transparency)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")

    # Blend the original image with the color mask using alpha compositing
    new_img = Image.blend(background, overlay, 0.2)

    return new_img


def display_overlay(test_image, prediction):
    """
    Display the overlay image showing the prediction on top of the test image.

    :param test_image: The test image as a numpy array.
    :param prediction: The prediction mask as a numpy array.
    """
    # Create the overlay image
    overlay_img = make_img_overlay(test_image, prediction)

    # Create a figure and display the overlay image
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay_img)
    plt.title('Overlay Image')
    plt.axis('off')  # Turn off the axis
    plt.show()


def display_multiple_overlays(test_images, predictions, grid_size=4):
    """
    Display overlay images showing the predictions on top of the test images in a grid format, 
    selecting images randomly.

    Args:
    test_images (list or array): A list or array of test images as numpy arrays.
    predictions (list or array): A list or array of prediction masks as numpy arrays.
    grid_size (int): The size of the grid to display (e.g., 4 for a 4x4 grid). Default is 4.
    """
    num_images = len(test_images)
    if num_images < grid_size ** 2:
        raise ValueError("Not enough images to fill the grid")

    # Select random indices
    selected_indices = np.random.choice(num_images, grid_size ** 2, replace=False)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = selected_indices[i * grid_size + j]
            overlay_img = make_img_overlay(test_images[idx], predictions[idx])
            axes[i, j].imshow(overlay_img)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


def display(display_list):
    """
    Display a list of images/masks in a single row.

    :param display_list: List of images/masks to display.
    """
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        image = display_list[i]
        
        # Display the image or mask appropriately based on its dimension
        if image.ndim == 2:  # Grayscale mask
            plt.imshow(image, cmap='gray')
        elif image.ndim == 3 and image.shape[-1] == 1:  # Single-channel mask
            plt.imshow(np.squeeze(image, axis=-1), cmap='gray')
        else:  # RGB Image
            plt.imshow(tf.keras.utils.array_to_img(image))
        
        plt.axis('off')
    plt.show()


def plot_training_validation_loss(history):
    """
    Plots the training and validation loss from a Keras model history.

    Args:
    history (History): The History object from Keras training.

    Saves the plot as 'train_validation_loss.png' and displays it.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 3))  # Set the figure size to 8x3 inches
    plt.plot(loss, 'r', label='Training loss')
    plt.plot(val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss', fontsize=16)  # Increase the title font size
    plt.xlabel('Epoch', fontsize=14)  # Increase the x-axis label font size
    plt.ylabel('Loss Value', fontsize=14)  # Increase the y-axis label font size
    plt.ylim([0, max(val_loss)*1.1])
    plt.legend(fontsize=12)  # Increase the legend font size
    plt.savefig("plots/train_validation_loss.png", dpi=300, bbox_inches='tight')  # Save the figure with higher resolution and remove extra whitespace
    plt.show()


def print_layer_details(model):
    """
    Prints important details for each layer in a Keras model in a table-like structure.

    Args:
    model (keras.Model): The Keras model whose layer details are to be printed.
    """
    # Define the header
    header = "{:<30} {:<20} {:<15}".format("Layer (Type)", "Attribute", "Value")
    print(header)
    print("-" * len(header))

    for layer in model.layers:
        layer_type = layer.__class__.__name__
        layer_name = layer.name
        config = layer.get_config()

        # Print layer name and type
        print("{:<30} {:<20} {:<15}".format(f"{layer_name} ({layer_type})", "", ""))

        # For convolutional layers
        if 'conv' in layer.name:
            details = [
                ("Filters", config.get('filters')),
                ("Kernel Size", config.get('kernel_size')),
                ("Strides", config.get('strides')),
                ("Padding", config.get('padding')),
                ("Activation", config.get('activation')),
                ("Use Bias", config.get('use_bias')),
            ]
            for attr, value in details:
                print("{:<30} {:<20} {:<15}".format("", attr, str(value)))

        # For dropout layers
        elif 'dropout' in layer.name:
            print("{:<30} {:<20} {:<15}".format("", "Dropout Rate", str(config.get('rate'))))

        # For pooling layers
        elif 'pool' in layer.name:
            details = [
                ("Pool Size", config.get('pool_size')),
                ("Strides", config.get('strides')),
                ("Padding", config.get('padding')),
            ]
            for attr, value in details:
                print("{:<30} {:<20} {:<15}".format("", attr, str(value)))

        # Add any other layer types and their relevant configurations as needed
        print("")


def save_model_with_history_info(model, history, info):
    """
    Saves the Keras model with a filename that includes the last loss, accuracy,
    validation loss, validation accuracy, and an additional info string.

    Args:
    model (keras.Model): The Keras model to be saved.
    history (History): The History object from Keras training.
    info (str): Additional information to include in the filename.
    """
    # Get the last values from the history
    last_loss = history.history['loss'][-1]
    last_accuracy = history.history['accuracy'][-1]
    last_val_loss = history.history['val_loss'][-1]
    last_val_accuracy = history.history['val_accuracy'][-1]

    # Save the model
    filename = f"models/model_{last_loss:.4f}_{last_accuracy:.4f}_{last_val_loss:.4f}_{last_val_accuracy:.4f}_{info}.keras"
    model.save(filename)
    print(f"Model saved as {filename}")