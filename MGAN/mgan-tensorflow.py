import os
import glob

import tensorflow as tf
import numpy as np
import helper


def generator(inputs, cond=None, reuse=False, is_training=True):
    with tf.variable_scope('generator', reuse=reuse):
        # set parameters
        w_init = tf.contrib.layers.xavier_initializer()
        # alpha = 0.2
        n_f = 64
        n_k = 3
        n_fc = 4096
        r_drop = 0.5
        r_nodrop = 0.0

        def encoder(e_in, n_filter, repeat=1):
            prev_in = e_in
            for ii in range(repeat):
                prev_in = tf.layers.conv2d(prev_in, n_filter, n_k, 1, 'same',
                                           kernel_initializer=w_init, activation=tf.nn.relu)
            e = tf.layers.conv2d(prev_in, n_filter, n_k, 2, 'same', kernel_initializer=w_init)
            e = tf.layers.batch_normalization(e, training=is_training)
            e = tf.nn.relu(e)
            return e

        def decoder(d_in, n_filter, drop_rate=0.0, repeat=1, last_activation=True):
            prev_in = d_in
            for ii in range(repeat):
                prev_in = tf.layers.conv2d_transpose(inputs=prev_in, filters=n_filter, kernel_size=n_k, strides=1,
                                                     padding='same')
                prev_in = tf.layers.dropout(prev_in, rate=drop_rate, training=True)
                prev_in = tf.nn.relu(prev_in)

            d = tf.layers.conv2d_transpose(inputs=prev_in, filters=n_filter, kernel_size=n_k, strides=2, padding='same')

            if last_activation:
                d = tf.layers.batch_normalization(d, training=is_training)
                d = tf.layers.dropout(d, rate=drop_rate, training=True)
                d = tf.nn.relu(d)
            else:
                d = tf.tanh(d)

            return d

        concated = inputs
        if cond is not None:
            concated = tf.concat([inputs, cond], axis=3)

        # encoder 1: [batch size, 512, 512, 6] ==> [batch size, 256, 256, 64]
        e1 = encoder(concated, n_f, repeat=1)

        # encoder 2: [batch size, 256, 256, 64] ==> [batch size, 128, 128, 128]
        e2 = encoder(e1, n_f * 2, repeat=1)

        # encoder 3: [batch size, 128, 128, 128] ==> [batch size, 64, 64, 256]
        e3 = encoder(e2, n_f * 4, repeat=1)

        # encoder 4: [batch size, 64, 64, 256] ==> [batch size, 32, 32, 512]
        e4 = encoder(e3, n_f * 8, repeat=2)

        # encoder 5: [batch size, 32, 32, 512] ==> [batch size, 16, 16, 512]
        e5 = encoder(e4, n_f * 8, repeat=2)

        # encoder 6: [batch size, 16, 16, 512] ==> [batch size, 8, 8, 512]
        e6 = encoder(e5, n_f * 8, repeat=2)

        # encoder 7: [batch size, 8, 8, 512] ==> [batch size, 4, 4, 512]
        e7 = encoder(e6, n_f * 8, repeat=2)

        # encoder 8: [batch size, 4, 4, 512] ==> [batch size, 2, 2, 512]
        e8 = encoder(e7, n_f * 8, repeat=2)

        # reshape e8: [batch size, 2, 2, 512] ==> [batch size, 2048]
        # dim: 2 * 2 * 512 = 2048
        e8_shape = e8.get_shape().as_list()
        dim = 1
        for d in e8_shape[1:]:
            dim *= d
        reshaped_e8 = tf.reshape(e8, [-1, dim])

        # dense 1: [batch size, 2048] ==> [batch size, 4096]
        dense1 = tf.layers.dense(reshaped_e8, n_fc, kernel_initializer=w_init, activation=tf.nn.relu)

        # dense 2: [batch size, 4096] ==> [batch size, 2048]
        dense2 = tf.layers.dense(dense1, dim, kernel_initializer=w_init, activation=tf.nn.relu)

        # reshape dense 2: [batch size, 2048] ==> [batch size, 2, 2, 512]
        reshaped_dense2 = tf.reshape(dense2, [-1, e8_shape[1], e8_shape[2], e8_shape[3]])

        # decoder 1: [batch size, 2, 2, 512 * 2] ==> [batch size, 4, 4, 512]
        concated_08 = tf.concat([reshaped_dense2, e8], axis=3)
        d1 = decoder(concated_08, n_f * 8, r_drop, repeat=2)

        # decoder 2: [batch size, 4, 4, 512 * 2] ==> [batch size, 8, 8, 512]
        concated_17 = tf.concat([d1, e7], axis=3)
        d2 = decoder(concated_17, n_f * 8, r_drop, repeat=2)

        # decoder 3: [batch size, 8, 8, 512 * 2] ==> [batch size, 16, 16, 512]
        concated_26 = tf.concat([d2, e6], axis=3)
        d3 = decoder(concated_26, n_f * 8, r_drop, repeat=2)

        # decoder 4: [batch size, 16, 16, 512 * 2] ==> [batch size, 32, 32, 512]
        concated_35 = tf.concat([d3, e5], axis=3)
        d4 = decoder(concated_35, n_f * 8, r_drop, repeat=2)

        # decoder 4: [batch size, 32, 32, 512 * 2] ==> [batch size, 64, 64, 256]
        concated_44 = tf.concat([d4, e4], axis=3)
        d5 = decoder(concated_44, n_f * 4, r_nodrop, repeat=1)

        # decoder 5: [batch size, 64, 64, 256 * 2] ==> [batch size, 128, 128, 128]
        concated_53 = tf.concat([d5, e3], axis=3)
        d6 = decoder(concated_53, n_f * 2, r_nodrop, repeat=1)

        # decoder 6: [batch size, 128, 128, 128 * 2] ==> [batch size, 256, 256, 64]
        concated_62 = tf.concat([d6, e2], axis=3)
        d7 = decoder(concated_62, n_f, r_nodrop, repeat=2)

        # decoder 13: [batch size, 256, 256, 64 * 2] ==> [batch size, 512, 512, 3]
        concated_71 = tf.concat([d7, e1], axis=3)
        d8 = decoder(concated_71, 3, r_nodrop, repeat=1, last_activation=False)

        return d8


def discriminator(inputs, reuse=False, is_training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        # set parameters
        w_init = tf.contrib.layers.xavier_initializer()
        alpha = 0.2
        n_f = 64
        n_k = 3
        n_fc = 512

        def encoder(e_in, n_filter, repeat=1):
            prev_in = e_in
            for ii in range(repeat):
                prev_in = tf.layers.conv2d(prev_in, n_filter, n_k, 1, 'same',
                                           kernel_initializer=w_init, activation=tf.nn.relu)
            e = tf.layers.conv2d(prev_in, n_filter, n_k, 2, 'same', kernel_initializer=w_init)
            e = tf.layers.batch_normalization(e, training=is_training)
            e = tf.maximum(alpha * e, e)
            return e

        # encoder 1: [batch size, 512, 512, 3] ==> [batch size, 256, 256, 64]
        e1 = encoder(inputs, n_f, repeat=1)

        # encoder 2: [batch size, 256, 256, 64] ==> [batch size, 128, 128, 128]
        e2 = encoder(e1, n_f * 2, repeat=1)

        # encoder 3: [batch size, 128, 128, 128] ==> [batch size, 64, 64, 256]
        e3 = encoder(e2, n_f * 4, repeat=1)

        # encoder 4: [batch size, 64, 64, 256] ==> [batch size, 32, 32, 512]
        e4 = encoder(e3, n_f * 8, repeat=2)

        # encoder 5: [batch size, 32, 32, 512] ==> [batch size, 16, 16, 512]
        e5 = encoder(e4, n_f * 8, repeat=2)

        # encoder 6: [batch size, 16, 16, 512] ==> [batch size, 8, 8, 512]
        e6 = encoder(e5, n_f * 8, repeat=2)

        # encoder 7: [batch size, 8, 8, 512] ==> [batch size, 4, 4, 512]
        e7 = encoder(e6, n_f * 8, repeat=2)

        # encoder 8: [batch size, 4, 4, 512] ==> [batch size, 2, 2, 512]
        e8 = encoder(e7, n_f * 8, repeat=2)

        # reshape e8: [batch size, 2, 2, 512] ==> [batch size, 2048]
        # dim: 2 * 2 * 512 = 2048
        e8_shape = e8.get_shape().as_list()
        dim = 1
        for d in e8_shape[1:]:
            dim *= d
        reshaped_e8 = tf.reshape(e8, [-1, dim])

        # dense 1: [batch size, 2048] ==> [batch size, 512]
        dense1 = tf.layers.dense(reshaped_e8, n_fc, kernel_initializer=w_init, activation=tf.nn.relu)

        # dense 2: [batch size, 512] ==> [batch size, 64]
        dense2 = tf.layers.dense(dense1, n_fc // 8, kernel_initializer=w_init, activation=tf.nn.relu)

        # dense 3: [batch size, 64] ==> [batch size, 1]
        dense3 = tf.layers.dense(dense2, 1, kernel_initializer=w_init, activation=tf.nn.relu)

        return dense3


class MGAN(object):
    def __init__(self):
        # parameters
        self.im_size = 512
        self.lmbda = 20.0
        self.beta1 = 0.5
        self.learning_rate = 0.00002

        # input place holders
        self.inputs_sketch = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3], name='inputs_sketch')
        self.inputs_real = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3], name='inputs_real')
        self.inputs_cond = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3], name='inputs_cond')

        # create generator & discriminator out
        self.gen_out = generator(self.inputs_sketch, self.inputs_cond, reuse=False, is_training=True)
        self.dis_logit_real = discriminator(self.inputs_real, reuse=False, is_training=True)
        self.dis_logit_fake = discriminator(self.gen_out, reuse=True, is_training=True)

        # model loss computation
        self.d_loss, self.g_loss = self.model_loss(self.dis_logit_real, self.dis_logit_fake,
                                                   self.inputs_real, self.gen_out)

        # model optimizer
        self.d_train_opt, self.g_train_opt = self.model_opt(self.d_loss, self.g_loss)



    def model_loss(self, dis_logit_real, dis_logit_fake, inputs_real, gen_out):
        # shorten cross entropy loss calculation
        def celoss_ones(logits):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))

        def celoss_zeros(logits):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)))

        d_loss_real = celoss_ones(dis_logit_real)
        d_loss_fake = celoss_ones(dis_logit_fake)
        d_loss = d_loss_real + d_loss_fake

        g_loss_l1 = tf.reduce_mean(tf.abs(inputs_real - gen_out))
        g_loss_gan = celoss_ones(dis_logit_fake)
        g_loss = self.lmbda * g_loss_l1 + g_loss_gan

        return d_loss, g_loss


    def model_opt(self, d_loss, g_loss):
        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimizers
        d_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt




def train(net, epochs, batch_size, train_input_dir, direction, print_every=30):
    steps = 0

    # prepare dataset
    train_dataset = helper.Dataset(train_input_dir, direction=direction, is_test=False)

    # use last image for testing
    for_testing = train_dataset.get_image_by_index(-1)
    test_a = [x for x, y, z in for_testing]
    test_b = [y for x, y, z in for_testing]
    test_c = [z for x, y, z in for_testing]
    test_a = np.array(test_a)
    test_b = np.array(test_b)
    test_c = np.array(test_c)

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

                fd = {
                    net.inputs_sketch: a,
                    net.inputs_real: b,
                    net.inputs_cond: c
                }

                _ = sess.run(net.d_train_opt, feed_dict=fd)
                _ = sess.run(net.g_train_opt, feed_dict=fd)
                _ = sess.run(net.g_train_opt, feed_dict=fd)

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = net.d_loss.eval(fd)
                    train_loss_g = net.g_loss.eval(fd)
                    train_loss_g_l1 = net.g_loss_l1.eval(fd)

                    print("Epoch {}/{}...".format(e + 1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g),
                          "L1 Loss: {:.4f}".format(train_loss_g_l1))

            save_fn = './assets/train-e-{:02d}.png'.format(e)
            test_out = sess.run(net.gen_out, feed_dict={net.inputs_sketch: test_a, net.inputs_cond: test_c})
            helper.save_result(save_fn, test_out, input_image=test_a, target_image=test_b)

        ckpt_fn = './checkpoints/MGAN.ckpt'
        saver.save(sess, ckpt_fn)

    return

def main():
    # prepare directories
    assets_dir = './assets/'
    ckpt_dir = './checkpoints/'
    if not os.path.isdir(assets_dir):
        os.mkdir(assets_dir)
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    epochs = 1
    batch_size = 1
    train_input_dir = 'D:\\db\\pixiv\\sketch-2-color\\merged_refined_512\\'
    direction = 'BtoA'

    mgan = MGAN()
    train(mgan, epochs, batch_size, train_input_dir, direction)
    
    return


if __name__ == '__main__':
    main()