import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# shorten cross entropy loss calculations
def celoss_ones(logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))


def celoss_zeros(logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)))


def get_perturbed_batch(minibatch):
    return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)


# mnist datset loader
def get_mnist(mnist_base_dir, mnist_type):
    if not (mnist_type == 'original-MNIST' or mnist_type == 'fashion-MNIST'):
        raise ValueError('Either "original-MNIST" or "fashion-MNIST"')

    mnist = input_data.read_data_sets(os.path.join(mnist_base_dir, mnist_type), one_hot=True)
    return mnist
