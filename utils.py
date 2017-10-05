import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import toimage


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


# gan validation function
def validation(val_out, val_block_size, image_fn, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image, mode=color_mode).save(image_fn)
