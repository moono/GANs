import os
import numpy as np
import tensorflow as tf
import time

import utils
import network
from dataset_loader import get_mnist_by_name
from losses import wgan_loss


class WGANGP(object):
    def __init__(self, name, dataset_type, gan_loss_type):
        # prepare directories
        self.assets_dir = './assets/{:s}'.format(name)
        self.ckpt_dir = './ckpts/{:s}'.format(name)
        self.ckpt_fn = os.path.join(self.ckpt_dir, '{:s}.ckpt'.format(name))
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        # setup variables
        self.dataset_type = dataset_type

        # tunable parameters
        self.z_dim = 100
        self.learning_rate = 1e-4
        self.epochs = 30
        self.batch_size = 128
        self.print_every = 30
        self.save_every = 5
        self.val_block_size = 10
        self.lmbd_gp = 10.0

        # start building graphs
        tf.reset_default_graph()

        # create placeholders
        self.running_bs = tf.placeholder(tf.int32, [], name='running_bs')
        self.latent_z = tf.placeholder(tf.float32, [None, self.z_dim], name='latent_z')
        self.real_images = tf.placeholder(tf.float32, [None, 28, 28, 1], name='real_images')

        # create generator & discriminator
        self.fake_images = network.generator(self.latent_z, is_training=True, use_bn=False)
        self.d_real_logits, _ = network.discriminator(self.real_images, is_training=True, use_bn=False)
        self.d_fake_logits, _ = network.discriminator(self.fake_images, is_training=True, use_bn=False)

        # compute model loss
        self.d_loss, self.g_loss = wgan_loss(self.d_real_logits, self.d_fake_logits)

        # add gradient penalty
        alpha = tf.random_uniform(shape=[self.running_bs, 1, 1, 1], minval=-1.0, maxval=1.0)
        interpolates = self.real_images + alpha * (self.fake_images - self.real_images)
        d_interpolates_logits, _ = network.discriminator(interpolates, is_training=True, use_bn=False)
        gradients = tf.gradients(d_interpolates_logits, [interpolates])[0]
        slopes = tf.sqrt(0.0001 + tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
        self.d_loss += self.lmbd_gp * gradient_penalty

        # prepare optimizers
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimize
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
        self.d_opt = optimizer.minimize(self.d_loss, var_list=d_vars)
        self.g_opt = optimizer.minimize(self.g_loss, var_list=g_vars, global_step=tf.train.get_or_create_global_step())

        # prepare saver for generator
        self.saver = tf.train.Saver(var_list=g_vars)
        return

    def train_step(self, sess, next_elem, steps, losses):
        # get real images
        elem = sess.run(next_elem)
        real_images = elem['image']
        batch_size = real_images.shape[0]

        # Sample random noise for G
        batch_z = np.random.uniform(-1.0, 1.0, size=(batch_size, self.z_dim))

        # Run optimizers
        feed_dict = {
            self.running_bs: batch_size,
            self.real_images: real_images,
            self.latent_z: batch_z
        }
        _, __ = sess.run([self.d_opt, self.g_opt], feed_dict=feed_dict)

        # print losses
        if steps % self.print_every == 0:
            # At the end of each epoch, get the losses and print them out
            train_loss_d = self.d_loss.eval(feed_dict)
            train_loss_g = self.g_loss.eval(feed_dict)

            print("Discriminator Loss: {:.4f}...".format(train_loss_d), "Generator Loss: {:.4f}".format(train_loss_g))
            losses.append((train_loss_d, train_loss_g))
        return

    def save_generator_output(self, sess, e, fixed_z):
        feed_dict = {self.latent_z: fixed_z}
        fake_out = sess.run(network.generator(self.latent_z, is_training=False, use_bn=False), feed_dict=feed_dict)
        image_fn = os.path.join(self.assets_dir,
                                '{:s}-e{:03d}.png'.format(self.dataset_type, e + 1))
        utils.validation(fake_out, self.val_block_size, image_fn, color_mode='L')
        return

    def train(self):
        # fix z for visualization
        n_fixed_samples = self.val_block_size * self.val_block_size
        fixed_z = np.random.uniform(-1.0, 1.0, size=(n_fixed_samples, self.z_dim))

        # get dataset
        mnist_dataset = get_mnist_by_name(self.batch_size, self.dataset_type)

        # setup tracking variables
        steps = 0
        losses = []

        start_time = time.time()

        with tf.Session() as sess:
            # reset tensorflow variables
            sess.run(tf.global_variables_initializer())

            # start training
            for e in range(self.epochs):
                # setup dataset iterator for graph mode
                iterator = mnist_dataset.make_one_shot_iterator()
                next_elem = iterator.get_next()

                while True:
                    try:
                        self.train_step(sess, next_elem, steps, losses)
                        steps += 1
                    except tf.errors.OutOfRangeError:
                        print('End of dataset')
                        break

                # save generation results at every n epochs
                if e % self.save_every == 0:
                    self.save_generator_output(sess, e, fixed_z)
                    self.saver.save(sess, self.ckpt_fn, global_step=tf.train.get_or_create_global_step())

            # save final output
            self.save_generator_output(sess, e, fixed_z)
            self.saver.save(sess, self.ckpt_fn, global_step=tf.train.get_or_create_global_step())

        end_time = time.time()
        elapsed_time = end_time - start_time

        # save losses as image
        losses_fn = os.path.join(self.assets_dir, '{:s}-losses.png'.format(self.dataset_type,))
        utils.save_losses(losses, ['Discriminator', 'Generator'], elapsed_time, losses_fn)
        return
