
import numpy as np
import random
import cv2
import scipy.ndimage
from PIL import Image, ImageEnhance
from skimage.transform import rotate


def elastic_deform(image, alpha, sigma):
    """
    Apply elastic deformation to the image.

    :param image: A numpy array of shape (H, W, C) representing the image.
    :param alpha: The scaling factor for deformation intensity.
    :param sigma: The standard deviation of the Gaussian filter used for smoothing.
    :return: Deformed image as a numpy array.
    """
    assert len(image.shape) == 3, "Image should be HxWxC"

    # Random displacement fields
    dx = scipy.ndimage.gaussian_filter((np.random.rand(*image.shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = scipy.ndimage.gaussian_filter((np.random.rand(*image.shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Create meshgrid for displacements
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    x, y = np.clip(x + dx, 0, image.shape[1] - 1), np.clip(y + dy, 0, image.shape[0] - 1)

    # Apply displacement
    deformed_image = np.zeros_like(image)
    for i in range(image.shape[2]):
        deformed_image[:, :, i] = scipy.ndimage.map_coordinates(image[:, :, i], [y, x], order=1)

    return deformed_image


def random_crop(img, mask, crop_size):
    """
    Randomly crop an image and its corresponding mask.

    :param img: The image to crop, a numpy array of shape (H, W, C).
    :param mask: The corresponding mask to crop, a numpy array of shape (H, W).
    :param crop_size: The size of the crop (height, width).
    
    :return: Cropped image and mask.
    """
    if (img.shape[0] < crop_size[0]) or (img.shape[1] < crop_size[1]):
        raise ValueError("Crop size must be smaller than image size")

    height, width = img.shape[0], img.shape[1]
    crop_height, crop_width = crop_size

    # Randomly choose the top-left corner of the cropping box
    x = random.randint(0, width - crop_width)
    y = random.randint(0, height - crop_height)

    cropped_img = img[y:y+crop_height, x:x+crop_width, :]
    cropped_mask = mask[y:y+crop_height, x:x+crop_width]

    return cropped_img, cropped_mask


def resize_and_pad(images, masks, new_height, new_width):
    """
    Resize images and masks to new dimensions, maintaining aspect ratio, and add padding if necessary.

    :param images: Numpy array of images.
    :param masks: Numpy array of masks.
    :param new_height: New height after resizing and padding.
    :param new_width: New width after resizing and padding.
    
    :return: Resized and padded images and masks.
    """
    resized_images = np.zeros((len(images), new_height, new_width, images.shape[3]))
    resized_masks = np.zeros((len(masks), new_height, new_width))

    for i, (img, mask) in enumerate(zip(images, masks)):
        # Calculate the aspect ratio and determine dimensions to maintain aspect ratio
        h, w = img.shape[:2]
        scale = min(new_width / w, new_height / h)
        scaled_w, scaled_h = int(w * scale), int(h * scale)

        # Resize image and mask
        resized_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

        # Calculate padding sizes
        pad_w = (new_width - scaled_w) // 2
        pad_h = (new_height - scaled_h) // 2

        # Pad resized image and mask
        resized_images[i] = np.pad(resized_img, ((pad_h, new_height - scaled_h - pad_h), (pad_w, new_width - scaled_w - pad_w), (0, 0)), mode='constant')
        resized_masks[i] = np.pad(resized_mask, ((pad_h, new_height - scaled_h - pad_h), (pad_w, new_width - scaled_w - pad_w)), mode='constant')

    return resized_images, resized_masks


def pad_to_size(img, target_size):
    """
    Pad an image to a target size.

    :param img: Image to pad.
    :param target_size: Target size as a tuple (height, width).
    :return: Padded image.
    """
    height, width = img.shape[:2]
    pad_height = (target_size[0] - height) // 2
    pad_width = (target_size[1] - width) // 2

    # Calculate padding for even and odd dimensions
    pad_height1, pad_height2 = pad_height, target_size[0] - height - pad_height
    pad_width1, pad_width2 = pad_width, target_size[1] - width - pad_width

    return np.pad(img, ((pad_height1, pad_height2), (pad_width1, pad_width2), (0, 0)), mode='constant')


def color_augmentation(image, enhancement_factors=(0.8, 1.2)):
    """
    Apply random color augmentation to an image.

    :param image: A PIL image.
    :param enhancement_factors: Tuple of min and max factors for augmentation.

    :return: Color-augmented PIL image.
    """
    # Randomly change the brightness, contrast, and saturation
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
    for enhancer in enhancers:
        factor = random.uniform(*enhancement_factors)
        image = enhancer(image).enhance(factor)

    return image


def scale_variation(images, masks, scale_factors):
    """
    Apply scale variation to images and masks.

    :param images: List or numpy array of images.
    :param masks: List or numpy array of corresponding masks.
    :param scale_factors: Tuple of min and max factors for augmentation.
    
    :return: Augmented images and masks.
    """
    augmented_images = []
    augmented_masks = []

    for img, mask in zip(images, masks):
        # Randomly choose a scale factor
        scale_factor = random.uniform(*scale_factors)

        # Resize image and mask
        scaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        scaled_mask = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        augmented_images.append(scaled_img)
        augmented_masks.append(scaled_mask)

    return np.array(augmented_images), np.array(augmented_masks)


def shearing(image, angle_x, angle_y):
    """
    Apply random shearing to an image.

    :param image: A PIL image.
    :param shear_range: Tuple of min and max shear angles in degrees.

    :return: Sheared PIL image.
    """
    #angle_x = random.uniform(*shear_range)
    #angle_y = random.uniform(*shear_range)
    #angle_x, angle_y = shear_range

    return image.transform(image.size, Image.AFFINE, (1, angle_x, 0, angle_y, 1, 0), resample=Image.BICUBIC)


def add_noise(image, noise_type, mean, std):
    """
    Add random noise to an image.

    :param image: A PIL image.
    :param noise_type: Type of noise to add.
    :param mean: Mean of the noise distribution.
    :param std: Standard deviation of the noise distribution.

    :return: Noisy PIL image.
    """
    image_array = np.array(image)

    if noise_type == "gaussian":
        # Generate Gaussian noise with the same shape as the image for each channel
        noise = np.random.normal(mean, std, image_array.shape[:-1])
        # Add the noise independently to each channel
        noisy_image_array = image_array + noise[:, :, np.newaxis]
    elif noise_type == "salt_and_pepper":
        # Generate salt-and-pepper noise with the same shape as the image for each channel
        noise = np.random.randint(0, 2, image_array.shape[:-1]) * 255
        # Add the noise independently to each channel
        noisy_image_array = image_array + noise[:, :, np.newaxis]
    else:
        raise ValueError("Invalid noise type")

    # Clip the values to be in the valid range [0, 255]
    noisy_image_array = np.clip(noisy_image_array, 0, 255)

    # Convert the noisy NumPy array back to a PIL image
    noisy_image = Image.fromarray(np.uint8(noisy_image_array))

    return noisy_image


# -------------------------------- Main augmentation function -------------------------------- #
def augment_data(images, masks, augment_options, original_size):
    """
    Apply data augmentation techniques to images and masks.

    :param images: List or numpy array of images.
    :param masks: List or numpy array of corresponding masks.
    :param augment_options: Dictionary with options for different augmentations.
    :param original_size: Original size of the images (height, width).
    
    :return: Augmented images and masks.
    """
    augmented_images = list(images)
    augmented_masks = list(masks)
    #augmented_images = []
    #augmented_masks = []

    # Color Augmentation
    if augment_options.get("color_augmentation"):
        print("Applying color augmentation")
        for img, mask in zip(images, masks):
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            color_augmented_img = color_augmentation(pil_img)
            augmented_images.append(np.array(color_augmented_img) / 255.0)
            augmented_masks.append(mask)  # No change in masks for color augmentation

    # Rotation
    if augment_options.get("rotation"):
        angles = augment_options["rotation"]["angles"]
        print(f"Applying rotation in {angles} angles")
        for img, mask in zip(images, masks):
            for angle in angles:
                augmented_images.append(rotate(img, angle, preserve_range=True))
                augmented_masks.append(rotate(mask, angle, preserve_range=True))

    # Elastic Deformation
    if augment_options.get("deformation"):
        print("Applying deformation")
        alpha, sigma = augment_options["deformation"]["alpha"], augment_options["deformation"]["sigma"]
        for img, mask in zip(images, masks):
            augmented_images.append(elastic_deform(img, alpha, sigma))
            augmented_masks.append(elastic_deform(mask, alpha, sigma))

    # Random Cropping and Padding
    if augment_options.get("cropping"):
        print("Applying cropping")
        crop_height, crop_width = augment_options["cropping"]["height"], augment_options["cropping"]["width"]
        for img, mask in zip(images, masks):
            cropped_img, cropped_mask = random_crop(img, mask, (crop_height, crop_width))
            padded_img = pad_to_size(cropped_img, original_size)
            padded_mask = pad_to_size(cropped_mask, original_size)
            augmented_images.append(padded_img)
            augmented_masks.append(padded_mask)

    # Flipping
    if augment_options.get("flipping"):
        print("Applying flipping")
        for img, mask in zip(images, masks):
            # Horizontal flip
            augmented_images.append(np.fliplr(img))
            augmented_masks.append(np.fliplr(mask))
            # Vertical flip
            augmented_images.append(np.flipud(img))
            augmented_masks.append(np.flipud(mask))
            
    # Scaling
    if augment_options.get("scaling"):
        print("Applying scaling")
        scale_factors = augment_options["scaling"]["factors"]
        for img, mask in zip(images, masks):
            scaled_img, scaled_mask = scale_variation([img], [mask], scale_factors)
            augmented_images.append(scaled_img)
            augmented_masks.append(scaled_mask)
            
    # Shearing
    if augment_options.get("shearing"):
        print("Applying shearing")
        shear_range = augment_options["shearing"]["range"]
        for img, mask in zip(images, masks):
            angle_x = random.uniform(*shear_range)
            angle_y = random.uniform(*shear_range)
            # Apply shearing to the image
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            sheared_img = shearing(pil_img, angle_x, angle_y)
            augmented_images.append(np.array(sheared_img) / 255.0)
            
            # Apply shearing to the mask as well
            pil_mask = Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8))
            sheared_mask = shearing(pil_mask, angle_x, angle_y)
            mask_array = np.array(sheared_mask) / 255.0
            augmented_masks.append(np.expand_dims(mask_array, axis=-1))
            
    # Noise
    if augment_options.get("noise"):
        print("Applying noise")
        noise_type = augment_options["noise"]["type"]
        mean, std = augment_options["noise"]["mean"], augment_options["noise"]["std"]
        for img, mask in zip(images, masks):
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            noisy_img = add_noise(pil_img, noise_type, mean, std)
            augmented_images.append(np.array(noisy_img) / 255.0)
            augmented_masks.append(mask)  # No change in masks for noise
    
    return np.array(augmented_images), np.array(augmented_masks)
