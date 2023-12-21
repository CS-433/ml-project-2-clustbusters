
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from kerastuner.tuners import Hyperband

from helper_functions import *
from augmentation_functions import *
from unet_functions import *
from prediction_functions import *


def main():

    root_dir = "training/"
    n = 100
    imgs, gt_imgs = load_dataset(root_dir, load_images_from_directory, max_images=100)

    # Constant parameters
    original_height, original_width = 400, 400      # Image sizes
    validation_size = 0                             # Validation set for patch threshold test
    test_size = 0.05                                # Test set for CNN fitting
    random_state = 42                               # Random seed for train/test/validation split
    threshold = 0.5                                 # Threshold for patch prediction

    # U-net hyperparameters
    activation='relu'
    depth = 4
    dropout_rate = 0.1
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    patience = 3
    batch_size = 10
    epochs = 25


    # Choose augmentation options
    augment_options = {
        "color_augmentation": False
    }

    # Apply augmentations
    augmented_imgs, augmented_gt_imgs = augment_data(imgs, gt_imgs, augment_options, (original_height, original_width))

    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(augmented_imgs, augmented_gt_imgs, test_size=test_size, random_state=random_state)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Make sure that the gt are binary 0 and 1
    y_train = (y_train > 0.5).astype(np.float32)
    y_test = (y_test > 0.5).astype(np.float32)

    model = models.load_model('models/model_0.0092_0.9881_0.0053_0.9925.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=25, epochs=3, callbacks=callbacks)

    
    test_root_dir = "test_set_images"
    test_images = load_images_from_subfolders(test_root_dir, n=50)
    predictions = predict_images(model, test_images)

    threshold=0.2
    submission_df = create_submission_entries(predictions, threshold=threshold)
    save_submission_to_csv(submission_df, 'submissions/sample_submission.csv')