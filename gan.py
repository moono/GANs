import os
import numpy as np
import tensorflow as tf
import time

import utils
import network


class GAN(object):
    def __init__(self, name, dataset_type, mnist_loader, epochs):
        # prepare directories
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
        self.g_out = network.generator(self.inputs_z, reuse=False, is_training=True)
        self.d_real_logits, _ = network.discriminator(self.inputs_x, reuse=False, is_training=True)
        self.d_fake_logits, _ = network.discriminator(self.g_out, reuse=True, is_training=True)

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
        losses = []

        start_time = time.time()

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
                        losses.append((train_loss_d, train_loss_g))

                    steps += 1

                # save generation results at every epochs
                if e % self.save_every == 0:
                    val_z = np.random.uniform(-1, 1, size=(val_size, self.z_dim))
                    val_out = sess.run(network.generator(self.inputs_z, reuse=True, is_training=False),
                                       feed_dict={self.inputs_z: val_z})
                    image_fn = os.path.join(self.assets_dir, '{:s}-val-e{:03d}.png'.format(self.dataset_type, e+1))
                    utils.validation(val_out, self.val_block_size, image_fn, color_mode='L')

        end_time = time.time()
        elapsed_time = end_time - start_time

        # save losses as image
        losses_fn = os.path.join(self.assets_dir, '{:s}-losses.png'.format(self.dataset_type))
        utils.save_losses(losses, ['Discriminator', 'Generator'], elapsed_time, losses_fn)
        return
