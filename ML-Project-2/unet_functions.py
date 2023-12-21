
from keras import layers, models, optimizers
from keras import backend as K


def build_unet_hp(hp):
    """
    Build a U-Net model with customizable parameters.

    :param img_height: Height of the input images.
    :param img_width: Width of the input images.
    :param img_channels: Number of channels in the input images.
    :param num_classes: Number of output classes.
    :param start_filters: Number of filters in the first layer.
    :param depth: Depth of the U-Net.
    :param dropout_rate: Dropout rate for regularization.
    :param activation: Activation function to use.
    
    :return: A Keras model representing the U-Net.
    """
    img_height = 400
    img_width  = 400
    img_channels = 3
    num_classes = 1
    depth = 4
    inputs = layers.Input((img_height, img_width, img_channels))
    activation = 'elu'
    
    hp_start_filters = hp.Int('start_filters', min_value=10, max_value=20, step=1)
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.25, step=0.03)
    hp_learning_rate = hp.Float('learning_rate', min_value=0.01, max_value=0.3, step=0.01)
    hp_kernel_size = hp.Choice('kernel_size', values=[3, 5, 7])
    hp_activation = hp.Choice('activation', values=['relu', 'leaky_relu', 'elu'])
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    hp_loss = hp.Choice('loss', values=['binary_crossentropy', 'mean_squared_error'])

    # Contraction path
    contraction_layers = []
    x = inputs
    for i in range(depth):
        x = layers.Conv2D(hp_start_filters * (2 ** i), (hp_kernel_size, hp_kernel_size), activation=hp_activation, kernel_initializer='he_normal', padding='same')(x)
        x = layers.Dropout(i * hp_dropout_rate)(x)
        x = layers.Conv2D(hp_start_filters * (2 ** i), (hp_kernel_size, hp_kernel_size), activation=activation, kernel_initializer='he_normal', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        contraction_layers.append(x)
        x = layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = layers.Conv2D(hp_start_filters * (2 ** depth), (hp_kernel_size, hp_kernel_size), activation=activation, kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(depth * hp_dropout_rate)(x)
    x = layers.Conv2D(hp_start_filters * (2 ** depth), (hp_kernel_size, hp_kernel_size), activation=activation, kernel_initializer='he_normal', padding='same')(x)

    # Expansive path
    for i in reversed(range(depth)):
        x = layers.Conv2DTranspose(hp_start_filters * (2 ** i), (hp_kernel_size, hp_kernel_size), strides=(2, 2), padding='same')(x)
        x = layers.concatenate([x, contraction_layers[i]])
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=get_optimizer(hp_optimizer, hp_learning_rate),
                metrics=['accuracy'], loss=hp_loss)

    return model

def get_optimizer(optimizer_name, learning_rate):
    if optimizer_name == 'adam':
        return optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        return optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        return optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))