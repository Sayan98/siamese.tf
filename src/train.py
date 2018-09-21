from model import network
import numpy as np
import tensorflow as tf
from utils import *


# hyperparameters

lr = 1e-3
beta1, beta2 = 0.9, 0.999
epochs = 50
batch_size = 256


# dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

x_train, x_test = x_train/255, x_test/255
input_dim = (28, 28, 1)


digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(x_test, digit_indices)


# model

base_network = network(input_shape=input_dim)

input_a = tf.keras.layers.Input(shape=(*input_dim,))
input_b = tf.keras.layers.Input(shape=(*input_dim,))

feat_a = base_network(input_a)
feat_b = base_network(input_b)

distance = tf.keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_a, feat_b])

mnist_model = tf.keras.Model(inputs=[input_a, input_b], outputs=[distance])

optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
metrics = ['accuracy']

mnist_model.compile(loss=contrastive_loss, optimizer=optim, metrics=metrics)


# train
visualize_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch,
                                                       logs: visualize_clusters(base_network.predict(x_test),
                                                                                y_test, epoch))

mnist_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[visualize_callback])


# test

pred = mnist_model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on test set: %0.4f%%' % (100 * te_acc))

