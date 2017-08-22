import tensorflow as tf
import numpy as np
import os
import time
import json

import helper


def batch_norm(x,  name="batch_norm"):
    eps = 1e-6
    with tf.variable_scope(name):
        nchannels = x.get_shape()[3]
        scale = tf.get_variable("scale", [nchannels], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        center = tf.get_variable("center", [nchannels], initializer=tf.constant_initializer(0.0, dtype = tf.float32))
        ave, dev = tf.nn.moments(x, axes=[1,2], keep_dims=True)
        inv_dev = tf.rsqrt(dev + eps)
        normalized = (x-ave)*inv_dev * scale + center
        return normalized

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        desired_shape = conv.get_shape().as_list()
        desired_shape[0] = 1
        conv = tf.reshape(tf.nn.bias_add(conv, biases), desired_shape)

        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        desired_shape = deconv.get_shape().as_list()
        desired_shape[0] = 1
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), desired_shape)

        return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def generator(postfix, inputs, out_channel, n_filter_start, alpha=0.2, stddev=0.02, reuse=False, is_training=True):
    scope_name = 'generator-{:s}'.format(postfix)
    with tf.variable_scope(scope_name, reuse=reuse):
        s = inputs.get_shape().as_list()[1]
        s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
            s / 64), int(s / 128)

        # imgs is (256 x 256 x input_c_dim)
        e1 = conv2d(inputs, n_filter_start, name=postfix + 'e1_conv')
        # e1 is (128 x 128 x n_filter_start)
        e2 = batch_norm(conv2d(lrelu(e1), n_filter_start * 2, name=postfix + 'e2_conv'), name=postfix + 'bn_e2')
        # e2 is (64 x 64 x n_filter_start*2)
        e3 = batch_norm(conv2d(lrelu(e2), n_filter_start * 4, name=postfix + 'e3_conv'), name=postfix + 'bn_e3')
        # e3 is (32 x 32 x n_filter_start*4)
        e4 = batch_norm(conv2d(lrelu(e3), n_filter_start * 8, name=postfix + 'e4_conv'), name=postfix + 'bn_e4')
        # e4 is (16 x 16 x n_filter_start*8)
        e5 = batch_norm(conv2d(lrelu(e4), n_filter_start * 8, name=postfix + 'e5_conv'), name=postfix + 'bn_e5')
        # e5 is (8 x 8 x n_filter_start*8)
        e6 = batch_norm(conv2d(lrelu(e5), n_filter_start * 8, name=postfix + 'e6_conv'), name=postfix + 'bn_e6')
        # e6 is (4 x 4 x n_filter_start*8)
        e7 = batch_norm(conv2d(lrelu(e6), n_filter_start * 8, name=postfix + 'e7_conv'), name=postfix + 'bn_e7')
        # e7 is (2 x 2 x n_filter_start*8)
        e8 = batch_norm(conv2d(lrelu(e7), n_filter_start * 8, name=postfix + 'e8_conv'), name=postfix + 'bn_e8')
        # e8 is (1 x 1 x n_filter_start*8)

        d1 = deconv2d(tf.nn.relu(e8), [1, s128, s128, n_filter_start * 8], name=postfix + 'd1')
        d1 = tf.nn.dropout(batch_norm(d1, name=postfix + 'bn_d1'), 0.5)
        d1 = tf.concat([d1, e7], 3)
        # d1 is (2 x 2 x n_filter_start*8*2)

        d2 = deconv2d(tf.nn.relu(d1), [1, s64, s64, n_filter_start * 8], name=postfix + 'd2')
        d2 = tf.nn.dropout(batch_norm(d2, name=postfix + 'bn_d2'), 0.5)

        d2 = tf.concat([d2, e6], 3)
        # d2 is (4 x 4 x n_filter_start*8*2)

        d3 = deconv2d(tf.nn.relu(d2), [1, s32, s32, n_filter_start * 8], name=postfix + 'd3')
        d3 = tf.nn.dropout(batch_norm(d3, name=postfix + 'bn_d3'), 0.5)

        d3 = tf.concat([d3, e5], 3)
        # d3 is (8 x 8 x n_filter_start*8*2)

        d4 = deconv2d(tf.nn.relu(d3), [1, s16, s16, n_filter_start * 8], name=postfix + 'd4')
        d4 = batch_norm(d4, name=postfix + 'bn_d4')

        d4 = tf.concat([d4, e4], 3)
        # d4 is (16 x 16 x n_filter_start*8*2)

        d5 = deconv2d(tf.nn.relu(d4), [1, s8, s8, n_filter_start * 4], name=postfix + 'd5')
        d5 = batch_norm(d5, name=postfix + 'bn_d5')
        d5 = tf.concat([d5, e3], 3)
        # d5 is (32 x 32 x n_filter_start*4*2)

        d6 = deconv2d(tf.nn.relu(d5), [1, s4, s4, n_filter_start * 2], name=postfix + 'd6')
        d6 = batch_norm(d6, name=postfix + 'bn_d6')
        d6 = tf.concat([d6, e2], 3)
        # d6 is (64 x 64 x n_filter_start*2*2)

        d7 = deconv2d(tf.nn.relu(d6), [1, s2, s2, n_filter_start], name=postfix + 'd7')
        d7 = batch_norm(d7, name=postfix + 'bn_d7')
        d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x n_filter_start*1*2)

        d8 = deconv2d(tf.nn.relu(d7), [1, s, s, out_channel], name=postfix + 'd8')
        # d8 is (256 x 256 x output_c_dim)
        return tf.nn.tanh(d8)

def discriminator(postfix, inputs, n_filter_start, alpha=0.2, stddev=0.02, reuse=False, is_training=True):
    scope_name = 'discriminator-{:s}'.format(postfix)
    with tf.variable_scope(scope_name, reuse=reuse):
        h0 = lrelu(conv2d(inputs, n_filter_start, name=postfix + 'h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(batch_norm(conv2d(h0, n_filter_start * 2, name=postfix + 'h1_conv'), name=postfix + 'bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(batch_norm(conv2d(h1, n_filter_start * 4, name=postfix + 'h2_conv'), name=postfix + 'bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(batch_norm(conv2d(h2, n_filter_start * 8, d_h=1, d_w=1, name=postfix + 'h3_conv'), name=postfix + 'bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, d_h=1, d_w=1, name=postfix + 'h4')
        return h4

def model_loss(input_u, input_v,
               g_model_u2v2u, g_model_v2u2v,
               d_model_u_fake_logits, d_model_u_real_logits, d_model_v_fake_logits, d_model_v_real_logits,
               lambda_u, lambda_v):
    # discriminator losses
    d_loss_real_u = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_u_real_logits,
                                                                           labels=tf.ones_like(d_model_u_real_logits)))
    d_loss_fake_u = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_u_fake_logits,
                                                                           labels=tf.zeros_like(d_model_u_fake_logits)))
    d_loss_u = d_loss_real_u + d_loss_fake_u

    d_loss_real_v = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_v_real_logits,
                                                                           labels=tf.ones_like(d_model_v_real_logits)))
    d_loss_fake_v = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_v_fake_logits,
                                                                           labels=tf.zeros_like(d_model_v_fake_logits)))
    d_loss_v = d_loss_real_v + d_loss_fake_v

    # generator losses
    g_loss_gan_u = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_u_fake_logits,
                                                                          labels=tf.ones_like(d_model_u_fake_logits)))
    g_loss_gan_v = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_model_v_fake_logits,
                                                                          labels=tf.ones_like(d_model_v_fake_logits)))
    # compute L1 loss of G
    g_loss_l1_u = tf.reduce_mean(tf.abs(g_model_u2v2u - input_u))
    g_loss_l1_v = tf.reduce_mean(tf.abs(g_model_v2u2v - input_v))

    g_loss_u = g_loss_gan_u + lambda_v * g_loss_l1_v
    g_loss_v = g_loss_gan_v + lambda_u * g_loss_l1_u

    d_loss = d_loss_u + d_loss_v
    g_loss = g_loss_u + g_loss_v

    return g_loss_l1_u, g_loss_l1_v, d_loss, g_loss

def model_opt(d_loss, g_loss, g_u2v_post_fix, g_v2u_post_fix, d_u_post_fix, d_v_post_fix, learning_rate):
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    # d_vars = [var for var in t_vars if var.name.startswith('discriminator-')]
    # g_vars = [var for var in t_vars if var.name.startswith('generator-')]
    u_d_vars = [var for var in t_vars if d_u_post_fix in var.name]
    v_d_vars = [var for var in t_vars if d_v_post_fix in var.name]
    u_g_vars = [var for var in t_vars if g_u2v_post_fix in var.name]
    v_g_vars = [var for var in t_vars if g_v2u_post_fix in var.name]
    d_vars = u_d_vars + v_d_vars
    g_vars = u_g_vars + v_g_vars

    print(len(d_vars))
    print(len(g_vars))

    # Optimize
    decay = 0.9
    d_train_opt = tf.train.RMSPropOptimizer(learning_rate, decay=decay).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.RMSPropOptimizer(learning_rate, decay=decay).minimize(g_loss, var_list=g_vars)
    # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #     g_train_opt = tf.train.RMSPropOptimizer(learning_rate, decay=decay).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

class DualGAN(object):
    def __init__(self, im_size, im_channel_u, im_channel_v):
        tf.reset_default_graph()

        #
        self.im_size, self.channel_u, self.channel_v = im_size, im_channel_u, im_channel_v

        self.n_g_filter_start = 64
        self.n_d_filter_start = 64
        self.alpha = 0.2
        self.stddev = 0.02

        # loss related
        self.beta1 = 0.5
        self.lambda_u = 20.0
        self.lambda_v = 20.0
        self.learning_rate = 0.00005

        # Build model
        self.input_u = tf.placeholder(tf.float32, [None, im_size, im_size, im_channel_u], name='input_u')
        self.input_v = tf.placeholder(tf.float32, [None, im_size, im_size, im_channel_v], name='input_v')

        # generators
        self.g_model_u2v = generator(postfix='Gu', inputs=self.input_u, out_channel=self.channel_v,
                                     n_filter_start=self.n_g_filter_start, alpha=self.alpha, stddev=self.stddev,
                                     reuse=False, is_training=True)
        self.g_model_v2u = generator(postfix='Gv', inputs=self.input_v, out_channel=self.channel_u,
                                     n_filter_start=self.n_g_filter_start, alpha=self.alpha, stddev=self.stddev,
                                     reuse=False, is_training=True)

        self.g_model_u2v2u = generator(postfix='Gv', inputs=self.g_model_u2v, out_channel=self.channel_u,
                                       n_filter_start=self.n_g_filter_start, alpha=self.alpha, stddev=self.stddev,
                                       reuse=True, is_training=True)
        self.g_model_v2u2v = generator(postfix='Gu', inputs=self.g_model_v2u, out_channel=self.channel_v,
                                       n_filter_start=self.n_g_filter_start, alpha=self.alpha, stddev=self.stddev,
                                       reuse=True, is_training=True)

        # discriminators
        self.d_model_u_real_logits = discriminator(postfix='Du', inputs=self.input_v,
                                                   n_filter_start=self.n_d_filter_start, alpha=self.alpha,
                                                   stddev=self.stddev, reuse=False, is_training=True)
        self.d_model_u_fake_logits = discriminator(postfix='Du', inputs=self.g_model_u2v,
                                                   n_filter_start=self.n_d_filter_start, alpha=self.alpha,
                                                   stddev=self.stddev, reuse=True, is_training=True)

        self.d_model_v_real_logits = discriminator(postfix='Dv', inputs=self.input_u,
                                                   n_filter_start=self.n_d_filter_start, alpha=self.alpha,
                                                   stddev=self.stddev, reuse=False, is_training=True)
        self.d_model_v_fake_logits = discriminator(postfix='Dv', inputs=self.g_model_v2u,
                                                   n_filter_start=self.n_d_filter_start, alpha=self.alpha,
                                                   stddev=self.stddev, reuse=True, is_training=True)

        # define loss & optimizer
        self.g_loss_l1_u, self.g_loss_l1_v, self.d_loss, self.g_loss = \
            model_loss(self.input_u, self.input_v,
                       self.g_model_u2v2u, self.g_model_v2u2v,
                       self.d_model_u_fake_logits, self.d_model_u_real_logits,
                       self.d_model_v_fake_logits, self.d_model_v_real_logits,
                       self.lambda_u, self.lambda_v)

        self.d_train_opt, self.g_train_opt = model_opt(self.d_loss, self.g_loss,
                                                       'Gu', 'Gv', 'Du', 'Dv',
                                                       self.learning_rate)

def train(net, dataset_name, train_data_loader, val_data_loader, epochs, batch_size, print_every=30, save_every=100):
    losses = []
    steps = 0

    # prepare saver for saving trained model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            # shuffle data randomly at every epoch
            train_data_loader.reset()
            # val_data_loader.reset()

            for ii in range(train_data_loader.n_images // batch_size):
                steps += 1

                batch_image_u, batch_image_v = train_data_loader.get_next_batch(batch_size)

                fd = {
                    net.input_u: batch_image_u,
                    net.input_v: batch_image_v
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
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))

                if steps % save_every == 0:
                    # save generated images on every epochs
                    random_index = np.random.randint(0, val_data_loader.n_images)
                    test_image_u, test_image_v = val_data_loader.get_image_by_index(random_index)
                    fd_val = {
                        net.input_u: test_image_u,
                        net.input_v: test_image_v
                    }
                    g_loss_l1_u, g_image_u_to_v_to_u, g_image_u_to_v = \
                        sess.run([net.g_loss_l1_u, net.g_model_u2v2u, net.g_model_u2v], feed_dict=fd_val)
                    g_loss_l1_v, g_image_v_to_u_to_v, g_image_v_to_u = \
                        sess.run([net.g_loss_l1_v, net.g_model_v2u2v, net.g_model_v2u], feed_dict=fd_val)

                    image_fn = './assets/{:s}/epoch_{:d}-{:d}_tf.png'.format(dataset_name, e, steps)
                    helper.save_result(image_fn,
                                       test_image_u, g_image_u_to_v, g_image_u_to_v_to_u,
                                       test_image_v, g_image_v_to_u, g_image_v_to_u_to_v)

        ckpt_fn = './checkpoints/DualGAN-{:s}.ckpt'.format(dataset_name)
        saver.save(sess, ckpt_fn)

    return losses

def test(net, dataset_name, val_data_loader):
    ckpt_fn = './checkpoints/DualGAN-{:s}.ckpt'.format(dataset_name)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_fn)

        # run on u set
        for ii in range(val_data_loader.n_images):
            test_image_u = val_data_loader.get_image_by_index_u(ii)

            g_image_u_to_v, g_image_u_to_v_to_u = sess.run([net.g_model_u2v, net.g_model_u2v2u],
                                                           feed_dict={net.input_u: test_image_u})

            image_fn = './assets/{:s}/{:s}_result_u_{:04d}_tf.png'.format(dataset_name, dataset_name, ii)
            helper.save_result_single_row(image_fn, test_image_u, g_image_u_to_v, g_image_u_to_v_to_u)

        # run on v set
        for ii in range(val_data_loader.n_images):
            test_image_v = val_data_loader.get_image_by_index_v(ii)
            g_image_v_to_u, g_image_v_to_u_to_v = sess.run([net.g_model_v2u, net.g_model_v2u2v],
                                                           feed_dict={net.input_v: test_image_v})

            image_fn = './assets/{:s}/{:s}_result_v_{:04d}_tf.png'.format(dataset_name, dataset_name, ii)
            helper.save_result_single_row(image_fn, test_image_v, g_image_v_to_u, g_image_v_to_u_to_v)

def main():
    # prepare directories
    assets_dir = './assets/'
    ckpt_dir = './checkpoints/'
    if not os.path.isdir(assets_dir):
        os.mkdir(assets_dir)
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    # parameters to run
    with open('./training_parameters.json') as json_data:
        parameter_set = json.load(json_data)

    # start working!!
    for param in parameter_set:
        fn_ext = param['file_extension']
        dataset_name = param['dataset_name']
        dataset_base_dir = param['dataset_base_dir']
        epochs = param['epochs']
        batch_size = param['batch_size']
        im_size = param['im_size']
        im_channel = param['im_channel']
        do_flip = param['do_flip']
        is_test = param['is_test']

        # make directory per dataset
        current_assets_dir = './assets/{}/'.format(dataset_name)
        if not os.path.isdir(current_assets_dir):
            os.mkdir(current_assets_dir)

        # set dataset folders
        train_dataset_dir_u = dataset_base_dir + '{:s}/train/A/'.format(dataset_name)
        train_dataset_dir_v = dataset_base_dir + '{:s}/train/B/'.format(dataset_name)
        val_dataset_dir_u = dataset_base_dir + '{:s}/val/A/'.format(dataset_name)
        val_dataset_dir_v = dataset_base_dir + '{:s}/val/B/'.format(dataset_name)

        # prepare network
        net = DualGAN(im_size=im_size, im_channel_u=im_channel, im_channel_v=im_channel)

        if not is_test:
            # load train & validation datasets
            train_data_loader = helper.Dataset(train_dataset_dir_u, train_dataset_dir_v, fn_ext,
                                               im_size, im_channel, im_channel, do_flip=do_flip, do_shuffle=True)
            val_data_loader = helper.Dataset(val_dataset_dir_u, val_dataset_dir_v, fn_ext,
                                             im_size, im_channel, im_channel, do_flip=False, do_shuffle=False)

            # start training
            start_time = time.time()
            losses = train(net, dataset_name, train_data_loader, val_data_loader, epochs, batch_size)
            end_time = time.time()
            total_time = end_time - start_time
            test_result_str = '[Training]: Data: {:s}, Epochs: {:3f}, Batch_size: {:2d}, Elapsed time: {:3f}\n'.format(
                dataset_name, epochs, batch_size, total_time)
            print(test_result_str)

            with open('./assets/test_summary.txt', 'a') as f:
                f.write(test_result_str)

        else:
            # load train datasets
            val_data_loader = helper.Dataset(val_dataset_dir_u, val_dataset_dir_v, fn_ext,
                                             im_size, im_channel, im_channel, do_flip=False, do_shuffle=False)

            # validation
            test(net, dataset_name, val_data_loader)

# def test1():
#     fn_ext = 'jpg'
#     dataset_name = 'sketch-photo'
#     val_dir_u = '../data_set/DualGAN/sketch-photo/val/A'
#     val_dir_v = '../data_set/DualGAN/sketch-photo/val/B'
#     im_size = 256
#     im_channel = 1
#
#     val_data_loader = helper.Dataset(val_dir_u, val_dir_v, fn_ext,
#                                      im_size, im_channel, im_channel, do_flip=False, do_shuffle=False)
#     net = DualGAN(im_size=im_size, im_channel_u=im_channel, im_channel_v=im_channel)
#     test(net, dataset_name, val_data_loader)

if __name__ == '__main__':
    main()
