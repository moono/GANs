import os
import numpy as np
import tensorflow as tf
import time

import utils
import network
from dataset_loader import get_mnist_by_name
from losses import gan_loss_v1, gan_loss_v2, auxilary_classifier_loss


class ACGAN(object):
    def __init__(self, name, dataset_type, gan_loss_type):
        # prepare directories
        self.assets_dir = './assets/{:s}'.format(name)
        self.ckpt_dir = './ckpts/{:s}'.format(name)
        self.ckpt_fn = os.path.join(self.ckpt_dir, '{:s}-{:s}.ckpt'.format(name, gan_loss_type))
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        # setup variables
        self.dataset_type = dataset_type
        self.gan_loss_type = gan_loss_type

        # tunable parameters
        self.y_dim = 10
        self.z_dim = 128
        self.learning_rate = 1e-4
        self.epochs = 30
        self.batch_size = 128
        self.print_every = 30
        self.save_every = 5
        self.val_block_size = 10

        # start building graphs
        tf.reset_default_graph()

        # create placeholders
        self.latent_z = tf.placeholder(tf.float32, [None, self.z_dim], name='latent_z')
        self.inputs_y = tf.placeholder(tf.float32, [None, self.y_dim], name='inputs_y')
        self.real_images = tf.placeholder(tf.float32, [None, 28, 28, 1], name='real_images')

        # create generator & discriminator
        self.fake_images = network.generator(self.latent_z, y=self.inputs_y, is_training=True, use_bn=True)
        self.d_real_logits, self.a_real_input = network.discriminator(self.real_images, y=self.inputs_y,
                                                                      is_training=True, use_bn=True)
        self.d_fake_logits, self.a_fake_input = network.discriminator(self.fake_images, y=self.inputs_y,
                                                                      is_training=True, use_bn=True)
        self.a_real_logits = network.classifier(self.a_real_input, self.y_dim, is_training=True, use_bn=True)
        self.a_fake_logits = network.classifier(self.a_fake_input, self.y_dim, is_training=True, use_bn=True)

        # compute model loss
        if gan_loss_type == 'v1':
            self.d_loss, self.g_loss = gan_loss_v1(self.d_real_logits, self.d_fake_logits)
        elif gan_loss_type == 'v2':
            self.d_loss, self.g_loss = gan_loss_v2(self.d_real_logits, self.d_fake_logits)
        else:
            raise ValueError('gan_loss_type must be either "v1" or "v2"!!')
        self.a_loss = auxilary_classifier_loss(self.a_real_logits, self.a_fake_logits, self.inputs_y)

        # prepare optimizers
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimize
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_opt = optimizer.minimize(self.d_loss, var_list=d_vars)
            self.g_opt = optimizer.minimize(self.g_loss, var_list=g_vars)
            self.a_opt = optimizer.minimize(self.a_loss, var_list=t_vars,
                                            global_step=tf.train.get_or_create_global_step())

        # prepare saver for generator
        self.saver = tf.train.Saver(var_list=g_vars)
        return

    def train_step(self, sess, next_elem, steps, losses):
        # get real images
        elem = sess.run(next_elem)
        real_images = elem['image']
        labels = elem['label']
        labels = np.eye(self.y_dim)[labels]
        batch_size = real_images.shape[0]

        # Sample random noise for G
        batch_z = np.random.uniform(-1.0, 1.0, size=(batch_size, self.z_dim))

        # Run optimizers
        feed_dict = {
            self.real_images: real_images,
            self.latent_z: batch_z,
            self.inputs_y: labels
        }
        _, __, ___ = sess.run([self.d_opt, self.g_opt, self.a_opt], feed_dict=feed_dict)

        # print losses
        if steps % self.print_every == 0:
            # At the end of each epoch, get the losses and print them out
            train_loss_d = self.d_loss.eval(feed_dict)
            train_loss_g = self.g_loss.eval(feed_dict)
            train_loss_a = self.a_loss.eval(feed_dict)

            print("Discriminator Loss: {:.4f}...".format(train_loss_d),
                  "Generator Loss: {:.4f}".format(train_loss_g),
                  "Auxilary Classifier Loss: {:.4f}...".format(train_loss_a))
            losses.append((train_loss_d, train_loss_g, train_loss_a))
        return

    def save_generator_output(self, sess, e, fixed_z, fixed_y):
        feed_dict = {self.latent_z: fixed_z, self.inputs_y: fixed_y}
        fake_out = sess.run(network.generator(self.latent_z, y=self.inputs_y, is_training=False, use_bn=True), feed_dict=feed_dict)
        image_fn = os.path.join(self.assets_dir,
                                '{:s}-{:s}-e{:03d}.png'.format(self.dataset_type, self.gan_loss_type, e + 1))
        utils.validation(fake_out, self.val_block_size, image_fn)
        return

    def train(self):
        # fix z & y for visualization
        n_fixed_samples = self.val_block_size * self.val_block_size
        fixed_z = np.random.uniform(-1.0, 1.0, size=(n_fixed_samples, self.z_dim))
        fixed_y = np.zeros(shape=[n_fixed_samples, self.y_dim], dtype=np.float32)
        for s in range(n_fixed_samples):
            loc = s % self.y_dim
            fixed_y[s, loc] = 1.0

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
                    self.save_generator_output(sess, e, fixed_z, fixed_y)
                    self.saver.save(sess, self.ckpt_fn, global_step=tf.train.get_or_create_global_step())

            # save final output
            self.save_generator_output(sess, e, fixed_z, fixed_y)
            self.saver.save(sess, self.ckpt_fn, global_step=tf.train.get_or_create_global_step())

        end_time = time.time()
        elapsed_time = end_time - start_time

        # save losses as image
        losses_fn = os.path.join(self.assets_dir, '{:s}-{:s}-losses.png'.format(self.dataset_type, self.gan_loss_type))
        utils.save_losses(losses, ['Discriminator', 'Generator', 'Auxilary'], elapsed_time, losses_fn)
        return
