import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf


def visualize_clusters(feat, labels, epoch):
    '''Visualise clusters
    '''
    print(f"Visualising clusters after epoch #{epoch+1}")

    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    f = plt.figure(figsize=(6,6))

    for j in range(10):
        plt.plot(feat[labels==j, 0].flatten(),
                 feat[labels==j, 1].flatten(),
                 '.', c=c[j], alpha=0.8)

    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    plt.savefig('clusters/epoch_%d.jpg' % (epoch + 1))



def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 0.5
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def euclidean_distance(vects):
    x, y = vects

    return tf.sqrt(tf.keras.backend.sum(tf.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1

    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]

            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]

            labels += [1, 0]

    return np.array(pairs), np.array(labels)
