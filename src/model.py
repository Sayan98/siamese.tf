import tensorflow as tf
from utils import contrastive_loss


def network(input_shape=None):
    """Siamese network model
    """
    model = tf.keras.Sequential([
                                 tf.keras.layers.Conv2D(32, (7, 7), input_shape=input_shape,
                                                        activation='relu', padding='same',
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                                 tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same'),

                                 tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same',
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                                 tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same'),

                                 tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                                 tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same'),

                                 tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                                 tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same'),

                                 tf.keras.layers.Conv2D(2, (1, 1), activation=None, padding='same',
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                                 tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same'),

                                 tf.keras.layers.Flatten()
                                 ])

    return model


if __name__ == "__main__":
    net = network(loss=contrastive_loss)

    tf.keras.utils.plot_model(net, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='LR')
