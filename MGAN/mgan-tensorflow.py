import os
import glob

import tensorflow as tf
import numpy as np
# import helper
from helpers.helper import Dataset, save_result
from helpers.ops import generator, discriminator


class MGAN(object):
    def __init__(self, minibatch_size=1):
        tf.reset_default_graph()

        # parameters
        self.im_size = 512
        self.lmbda_l1 = 20.0
        self.lmbda_gp = 10.0
        self.mb_size = minibatch_size
        self.beta1 = 0.5
        self.learning_rate = 0.0002

        # input place holders
        self.inputs_sketch = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3], name='inputs_sketch')
        self.inputs_real = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3], name='inputs_real')
        self.inputs_cond = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3], name='inputs_cond')
        self.inputs_pt = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3], name='inputs_perturbed')

        # create generator & discriminator out
        self.gen_out = generator(self.inputs_sketch, self.inputs_cond, reuse=False, is_training=True)
        self.dis_logit_real = discriminator(self.inputs_sketch, self.inputs_real, reuse=False, is_training=True)
        self.dis_logit_fake = discriminator(self.inputs_sketch, self.gen_out, reuse=True, is_training=True)

        # model loss computation
        self.d_loss, self.g_loss = self.model_loss(self.dis_logit_real, self.dis_logit_fake,
                                                   self.inputs_sketch, self.inputs_real, self.gen_out, self.inputs_pt)

        # model optimizer
        self.d_train_opt, self.g_train_opt = self.model_opt(self.d_loss, self.g_loss)



    def model_loss(self, dis_logit_real, dis_logit_fake, inputs_sketch, inputs_real, gen_out, inputs_pt):
        # shorten cross entropy loss calculation
        def celoss_ones(logits):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))

        def celoss_zeros(logits):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)))

        # discriminator losses
        d_loss_real = celoss_ones(dis_logit_real)
        d_loss_fake = celoss_zeros(dis_logit_fake)

        # compute gradient penalty
        alpha_t = tf.random_uniform(shape=[self.mb_size, 1], minval=0., maxval=1.)
        differences_t = inputs_pt - inputs_real
        interpolated_t = inputs_real + (alpha_t * differences_t)
        gradients = tf.gradients(discriminator(inputs_sketch, interpolated_t, reuse=True, is_training=True),
                                 [interpolated_t])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        d_loss = d_loss_real + d_loss_fake + self.lmbda_gp * gradient_penalty

        # generator losses
        g_loss_gan = celoss_ones(dis_logit_fake)
        g_loss_l1 = tf.reduce_mean(tf.abs(inputs_real - gen_out))
        g_loss = self.lmbda_l1 * g_loss_l1 + g_loss_gan

        return d_loss, g_loss


    def model_opt(self, d_loss, g_loss):
        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt


def test_single_image(data_loader, save_index, sess, net):
    index = np.random.randint(0, data_loader.n_images)

    # use last image for testing
    for_testing = data_loader.get_image_by_index(index)
    test_a = [x for x, y, z in for_testing]
    test_b = [y for x, y, z in for_testing]
    test_c = [z for x, y, z in for_testing]
    test_a = np.array(test_a)
    test_b = np.array(test_b)
    test_c = np.array(test_c)

    save_fn = './assets/train-out{:02d}.png'.format(save_index)
    test_out = sess.run(net.gen_out, feed_dict={net.inputs_sketch: test_a, net.inputs_cond: test_c})
    save_result(save_fn, test_out, input_image=test_a, target_image=test_b)
    return


# input preturbation function
def get_perturbed_batch(minibatch):
    return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)


def train(net, epochs, batch_size, train_input_dir, direction, print_every=30, save_every=50):
    steps = 0

    # prepare dataset
    train_dataset = Dataset(train_input_dir, direction=direction, is_test=False)

    # prepare saver for saving trained model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for ii in range(train_dataset.n_images//batch_size):
                steps += 1

                # will return list of tuples [ (sketch, color, color_hint), ... , (sketch, color, color_hint)]
                batch_images_tuple = train_dataset.get_next_batch(batch_size)

                a = [x for x, y, z in batch_images_tuple]
                b = [y for x, y, z in batch_images_tuple]
                c = [z for x, y, z in batch_images_tuple]
                a = np.array(a)
                b = np.array(b)
                c = np.array(c)

                # create perturbed target
                p = get_perturbed_batch(b)

                fd = {
                    net.inputs_sketch: a,
                    net.inputs_real: b,
                    net.inputs_cond: c,
                    net.inputs_pt: p
                }

                _ = sess.run(net.d_train_opt, feed_dict=fd)
                _ = sess.run(net.g_train_opt, feed_dict=fd)
                _ = sess.run(net.g_train_opt, feed_dict=fd)

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = net.d_loss.eval(fd)
                    train_loss_g = net.g_loss.eval(fd)

                    print("Epoch {}/{}...".format(e + 1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

            # save train output
            test_single_image(train_dataset, e, sess, net)

            # save trained model
            if e % save_every == 0:
                saver.save(sess, './checkpoints/MGAN.ckpt', global_step=steps)

        saver.save(sess, './checkpoints/MGAN.ckpt')
    return


def main():
    # prepare directories
    assets_dir = './assets/'
    ckpt_dir = './checkpoints/'
    if not os.path.isdir(assets_dir):
        os.mkdir(assets_dir)
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    epochs = 50
    batch_size = 1
    train_input_dir = 'd:/db/getchu/merged_512/'
    direction = 'BtoA'

    mgan = MGAN()
    train(mgan, epochs, batch_size, train_input_dir, direction)
    
    return


if __name__ == '__main__':
    main()