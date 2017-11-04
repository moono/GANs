import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

class DRAGAN(object):
    def __init__(self, minibatch_size=128):
        tf.reset_default_graph()

        # input image is MNIST image with vectorized 28 * 28 = 784
        self.x_dim = 784
        self.z_dim = 100
        self.h_dim = 128
        self.lambd = 10.0
        self.learning_rate = 0.001
        self.beta1 = 0.5
        self.mb_size = minibatch_size

        # create placeholders
        self.inputs_x = tf.placeholder(tf.float32, shape=[None, self.x_dim], name='inputs_x')
        self.inputs_z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='inputs_z')
        self.inputs_p = tf.placeholder(tf.float32, shape=[None, self.x_dim], name='inputs_p')

        # create generator & discriminators
        self.g_model = self.generator(self.inputs_z, reuse=False)
        self.d_real_logits = self.discriminator(self.inputs_x, reuse=False)
        self.d_fake_logits = self.discriminator(self.g_model, reuse=True)
        self.d_purtubed_logits = self.discriminator(self.inputs_p, reuse=True)

        # model loss
        self.d_loss, self.g_loss, self.gradient_penalty = self.model_loss(self.d_real_logits,
                                                                          self.d_fake_logits,
                                                                          self.inputs_x,
                                                                          self.inputs_p)

        # optimizer
        self.d_train_opt, self.g_train_opt = self.model_opt(self.d_loss, self.g_loss)


    def generator(self, inputs, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            w_init = tf.contrib.layers.xavier_initializer()

            l1 = tf.layers.dense(inputs, units=self.h_dim, activation=None, use_bias=True, kernel_initializer=w_init)
            l1 = tf.nn.relu(l1)

            logits = tf.layers.dense(l1, units=self.x_dim, activation=None, use_bias=True, kernel_initializer=w_init)
            out = tf.nn.tanh(logits)

            return out

    def discriminator(self, inputs, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            w_init = tf.contrib.layers.xavier_initializer()

            l1 = tf.layers.dense(inputs, units=self.h_dim, activation=None, use_bias=True, kernel_initializer=w_init)
            l1 = tf.nn.relu(l1)

            logits = tf.layers.dense(l1, units=1, activation=None, use_bias=True, kernel_initializer=w_init)

            return logits


    def model_loss(self, d_real_logits, d_fake_logits, inputs_x, inputs_p):
        # shorten cross entropy loss calculation
        def celoss_ones(logits):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))

        def celoss_zeros(logits):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)))

        # discriminator loss
        d_real_loss = celoss_ones(d_real_logits)
        d_fake_loss = celoss_zeros(d_fake_logits)
        d_loss = d_real_loss + d_fake_loss

        # generator loss &
        g_loss = celoss_ones(d_fake_logits)

        # compute gradient penalty
        alpha = tf.random_uniform(shape=[self.mb_size, 1], minval=0., maxval=1.)
        differences = inputs_p - inputs_x
        interpolated = inputs_x + (alpha * differences)
        gradients = tf.gradients(self.discriminator(interpolated, reuse=True), [interpolated])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        # update d_loss
        d_loss += (self.lambd * gradient_penalty)

        return d_loss, g_loss, gradient_penalty

    def model_opt(self, d_loss, g_loss):
        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimizers
        d_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt

# image save function
def save_generator_output(samples, img_str, title):
    n_rows, n_cols = 5, 5

    reshaped = np.reshape(samples, newshape=[n_rows, n_cols, -1])

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5,5), sharey=True, sharex=True)
    for ax_row, img_row in zip(axes, reshaped):
        for ax, img in zip(ax_row, img_row):
            ax.axis('off')
            ax.set_adjustable('box-forced')
            rescaled = ((img + 1.0) * 127.5).astype(np.uint8)
            ax.imshow(rescaled.reshape((28,28)), cmap='Greys_r', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
    plt.savefig(img_str)
    plt.close(fig)

def get_perturbed_batch(minibatch):
    return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)

def train(net, epochs, batch_size, x_size, z_size, print_every=50):
    # get data sets
    mnist = input_data.read_data_sets('../data_set/MNIST_data', one_hot=True)

    fixed_z = np.random.uniform(-1, 1, size=(25, z_size))

    steps = 0
    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for ii in range(mnist.train.num_examples // batch_size):
                steps += 1

                # get batches
                x_, _ = mnist.train.next_batch(batch_size)

                # Get images rescale to pass to D
                x_ = x_.reshape((batch_size, x_size))
                x_ = x_ * 2 - 1

                # Sample random noise for G
                z_ = np.random.uniform(-1, 1, size=(batch_size, z_size))

                # create perturbed x
                p_ = get_perturbed_batch(x_)

                # Run optimizers
                sess.run(net.d_train_opt, feed_dict={net.inputs_x: x_, net.inputs_z: z_, net.inputs_p: p_})
                sess.run(net.g_train_opt, feed_dict={net.inputs_x: x_, net.inputs_z: z_, net.inputs_p: p_})

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d, train_loss_g, gradient_penalty = \
                        sess.run([net.d_loss, net.g_loss, net.gradient_penalty],
                                 feed_dict={net.inputs_x: x_, net.inputs_z: z_, net.inputs_p: p_})


                    print("Epoch {}/{}...".format(e + 1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g),
                          "Gradient Penalty: {:.4f}".format(gradient_penalty))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))

            # save generated images on every epochs
            image_fn = './assets/epoch_{:d}_tf.png'.format(e)
            image_title = 'epoch {:d}'.format(e)
            samples = sess.run(net.generator(net.inputs_z, reuse=True), feed_dict={net.inputs_z: fixed_z})
            save_generator_output(samples, image_fn, image_title)


    return losses

def main():
    # prepare directories
    assets_dir = './assets/'
    if not os.path.isdir(assets_dir):
        os.mkdir(assets_dir)

    epochs = 30
    batch_size = 128

    # create dragan network
    net = DRAGAN(batch_size)

    # start training
    start_time = time.time()
    losses = train(net, epochs=epochs, batch_size=batch_size, x_size=net.x_dim, z_size=net.z_dim)
    end_time = time.time()
    total_time = end_time - start_time
    test_result_str = '[Training]: Epochs: {:3f}, Batch_size: {:2d}, Elapsed time: {:3f}\n'\
        .format(epochs, batch_size, total_time)

    print(test_result_str)

    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('./assets/losses_tf.png')
    plt.close(fig)


if __name__ == '__main__':
    main()