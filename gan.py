import os
import numpy as np
import tensorflow as tf
import json
from pprint import pprint

import utils


def generator(z, reuse=False, is_training=True):
    with tf.variable_scope('generator', reuse=reuse):
        alpha = 0.2
        n_filter = 512
        n_kernel = 4
        w_init = tf.contrib.layers.xavier_initializer()

        # 1. reshape z-vector to fit as 2d shape image with fully connected layer
        l1 = tf.layers.dense(z, units=3 * 3 * n_filter, kernel_initializer=w_init)
        l1 = tf.reshape(l1, shape=[-1, 3, 3, n_filter])
        l1 = tf.maximum(alpha * l1, l1)

        # 2. layer2 - [batch size, 3, 3, 512] ==> [batch size, 7, 7, 512]
        l2 = tf.layers.conv2d_transpose(l1, filters=n_filter // 2, kernel_size=3, strides=2, padding='valid',
                                        kernel_initializer=w_init)
        l2 = tf.layers.batch_normalization(l2, training=is_training)
        l2 = tf.maximum(alpha * l2, l2)

        # 3. layer3 - [batch size, 7, 7, 256] ==> [batch size, 14, 14, 128]
        l3 = tf.layers.conv2d_transpose(l2, filters=n_filter // 4, kernel_size=n_kernel, strides=2, padding='same',
                                        kernel_initializer=w_init)
        l3 = tf.layers.batch_normalization(l3, training=is_training)
        l3 = tf.maximum(alpha * l3, l3)

        # 4. layer4 - [batch size, 14, 14, 128] ==> [batch size, 28, 28, 1]
        l4 = tf.layers.conv2d_transpose(l3, filters=1, kernel_size=n_kernel, strides=2, padding='same',
                                        kernel_initializer=w_init)
        out = tf.tanh(l4)
        return out


def discriminator(x, reuse=False, is_training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        alpha = 0.2
        n_filter = 64
        n_kernel = 4
        w_init = tf.contrib.layers.xavier_initializer()

        # 1. layer 1 - [batch size, 28, 28, 1] ==> [batch size, 14, 14, 64]
        l1 = tf.layers.conv2d(x, filters=n_filter, kernel_size=n_kernel, strides=2, padding='same',
                              kernel_initializer=w_init)
        l1 = tf.maximum(alpha * l1, l1)

        # 2. layer 2 - [batch size, 14, 14, 64] ==> [batch size, 7, 7, 128]
        l2 = tf.layers.conv2d(l1, filters=n_filter * 2, kernel_size=n_kernel, strides=2, padding='same',
                              kernel_initializer=w_init)
        l2 = tf.layers.batch_normalization(l2, training=is_training)
        l2 = tf.maximum(alpha * l2, l2)

        # 3. layer 3 - [batch size, 7, 7, 128] ==> [batch size, 4, 4, 256]
        l3 = tf.layers.conv2d(l2, filters=n_filter * 4, kernel_size=n_kernel, strides=2, padding='same',
                              kernel_initializer=w_init)
        l3 = tf.layers.batch_normalization(l3, training=is_training)
        l3 = tf.maximum(alpha * l3, l3)

        # 4. flatten layer & and finalize
        l4 = tf.reshape(l3, shape=[-1, 4, 4, 256])
        l4 = tf.layers.dense(l4, units=1, kernel_initializer=w_init)
        return l4


class GAN(object):
    def __init__(self, dataset_type, mnist_loader, epochs):
        # prepare directories
        name = 'vanilla-gan'
        self.assets_dir = './assets/{:s}'.format(name)
        self.ckpt_dir = './checkpoints/{:s}'.format(name)
        if not os.path.isdir(self.assets_dir):
            os.makedirs(self.assets_dir)
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        #
        self.dataset_type = dataset_type
        self.mnist_loader = mnist_loader

        # tunable parameters
        self.z_dim = 100
        self.learning_rate = 0.0002
        self.epochs = epochs
        self.batch_size = 128
        self.print_every = 30
        self.save_every = 1
        self.val_block_size = 10

        # start building graphs
        tf.reset_default_graph()

        # create placeholders
        self.inputs_x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='inputs_x')
        self.inputs_z = tf.placeholder(tf.float32, [None, self.z_dim], name='inputs_z')

        # create generator & discriminator
        self.g_out = generator(self.inputs_z, reuse=False, is_training=True)
        self.d_real_logits = discriminator(self.inputs_x, reuse=False, is_training=True)
        self.d_fake_logits = discriminator(self.g_out, reuse=True, is_training=True)

        # compute model loss
        self.d_loss, self.g_loss = self.model_loss(self.d_real_logits, self.d_fake_logits)

        # model optimizer
        self.d_opt, self.g_opt = self.model_opt(self.d_loss, self.g_loss)
        return

    @ staticmethod
    def model_loss(d_real_logits, d_fake_logits):
        # discriminator loss
        d_loss_real = utils.celoss_ones(d_real_logits)
        d_loss_fake = utils.celoss_zeros(d_fake_logits)
        d_loss = d_loss_real + d_loss_fake

        # generator loss
        g_loss = utils.celoss_ones(d_fake_logits)
        return d_loss, g_loss

    def model_opt(self, d_loss, g_loss):
        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimize
        beta1 = 0.5
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt

    def train(self):
        val_size = self.val_block_size * self.val_block_size
        steps = 0

        with tf.Session() as sess:
            # reset tensorflow variables
            sess.run(tf.global_variables_initializer())

            # start training
            for e in range(self.epochs):
                for ii in range(self.mnist_loader.train.num_examples // self.batch_size):
                    # no need labels
                    batch_x, _ = self.mnist_loader.train.next_batch(self.batch_size)

                    # rescale images to -1 ~ 1
                    batch_x = np.reshape(batch_x, (-1, 28, 28, 1))
                    batch_x = batch_x * 2.0 - 1.0

                    # Sample random noise for G
                    batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

                    # Run optimizers
                    _ = sess.run(self.d_opt, feed_dict={self.inputs_x: batch_x, self.inputs_z: batch_z})
                    _ = sess.run(self.g_opt, feed_dict={self.inputs_x: batch_x, self.inputs_z: batch_z})

                    # print losses
                    if steps % self.print_every == 0:
                        # At the end of each epoch, get the losses and print them out
                        train_loss_d = self.d_loss.eval({self.inputs_x: batch_x, self.inputs_z: batch_z})
                        train_loss_g = self.g_loss.eval({self.inputs_z: batch_z})

                        print("Epoch {}/{}...".format(e + 1, self.epochs),
                              "Discriminator Loss: {:.4f}...".format(train_loss_d),
                              "Generator Loss: {:.4f}".format(train_loss_g))
                    steps += 1

                # save generation results at every epochs
                if e % self.save_every == 0:
                    val_z = np.random.uniform(-1, 1, size=(val_size, self.z_dim))
                    val_out = sess.run(generator(self.inputs_z, reuse=True, is_training=False),
                                       feed_dict={self.inputs_z: val_z})
                    image_fn = os.path.join(self.assets_dir, '{:s}-val-e{:03d}.png'.format(self.dataset_type, e+1))
                    self.validation(val_out, image_fn, color_mode='L')
        return

    def validation(self, val_out, image_fn, color_mode):
        from scipy.misc import toimage

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
            if (b+1) % self.val_block_size == 0:
                if final_image.size == 0:
                    final_image = single_row
                else:
                    final_image = np.concatenate((final_image, single_row), axis=0)

                # reset single row
                single_row = np.array([])

        if final_image.shape[2] == 1:
            final_image = np.squeeze(final_image, axis=2)
        toimage(final_image, mode=color_mode).save(image_fn)


def main():
    # get training parameters
    with open('params.json') as f:
        gan_params = json.load(f)

    model_name = 'GAN'
    print('--{:s} params--'.format(model_name))
    pprint(gan_params)

    dataset_base_dir = './data_set'
    for param in gan_params:
        epochs = int(param["epochs"])
        mnist_type = param["mnist-type"]
        mnist = utils.get_mnist(dataset_base_dir, mnist_type)

        print('Training {:s} with epochs: {:d}, dataset: {:s}'.format(model_name, epochs, mnist_type))
        net = GAN(mnist_type, mnist, epochs)
        net.train()

    return


if __name__ == '__main__':
    main()
