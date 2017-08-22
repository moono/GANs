import tensorflow as tf
# import numpy as np
import os
import time
import json

import helper


def generator(postfix, inputs, out_channel, n_filter_start, alpha=0.2, stddev=0.02, reuse=False, is_training=True):
    scope_name = 'generator-{:s}'.format(postfix)
    with tf.variable_scope(scope_name, reuse=reuse):
        w_init_encoder = tf.truncated_normal_initializer(stddev=stddev)
        w_init_decoder = tf.random_normal_initializer(stddev=stddev)
        use_bias = True

        # prepare to stack layers to follow U-Net shape
        # inputs -> e1 -> e2 -> e3 -> e4 -> e5 -> e6 -> e7
        #           |     |     |     |     |     |     |  \
        #           |     |     |     |     |     |     |   e8
        #           V     V     V     V     V     V     V  /
        #     d8 <- d7 <- d6 <- d5 <- d4 <- d3 <- d2 <- d1
        layers = []

        # define names for each tensor operations
        ecnames = []     # encoder convolution
        ecbnnames = []   # encoder batch-normalization
        dcnames = []     # decoder deconvolution
        dcbnnames = []   # decoder batch-normalization
        dcdnames = []    # decoder dropout
        for ii in range(1, 9):
            # encoders
            ecname = 'g_ec{:d}_{:s}'.format(ii, postfix)
            ecbnname = 'g_ecbn{:d}_{:s}'.format(ii, postfix)
            ecnames.append(ecname)
            ecbnnames.append(ecbnname)

            # decoders
            dcname = 'g_dc{:d}_{:s}'.format(ii, postfix)
            dcbnname = 'g_dcbn{:d}_{:s}'.format(ii, postfix)
            dcdname = 'g_dcd{:d}_{:s}'.format(ii, postfix)
            dcnames.append(dcname)
            dcbnnames.append(dcbnname)
            dcdnames.append(dcdname)

        # expected inputs shape: [batch size, 256, 256, input_channel]

        # encoders
        # make [batch size, 128, 128, 64]
        encoder1 = tf.layers.conv2d(inputs, filters=n_filter_start, kernel_size=5, strides=2, padding='same',
                                    kernel_initializer=w_init_encoder, use_bias=use_bias, name=ecnames[0])
        layers.append(encoder1)

        encoder_spec = [
            n_filter_start * 2,  # encoder 2: [batch size, 128, 128, 64] => [batch size, 64, 64, 128]
            n_filter_start * 4,  # encoder 3: [batch size, 64, 64, 128] => [batch size, 32, 32, 256]
            n_filter_start * 8,  # encoder 4: [batch size, 32, 32, 256] => [batch size, 16, 16, 512]
            n_filter_start * 8,  # encoder 5: [batch size, 16, 16, 512] => [batch size, 8, 8, 512]
            n_filter_start * 8,  # encoder 6: [batch size, 8, 8, 512] => [batch size, 4, 4, 512]
            n_filter_start * 8,  # encoder 7: [batch size, 4, 4, 512] => [batch size, 2, 2, 512]
            n_filter_start * 8,  # encoder 8: [batch size, 2, 2, 512] => [batch size, 1, 1, 512]
        ]

        for ii, n_filters in enumerate(encoder_spec):
            prev_activated = tf.maximum(alpha * layers[-1], layers[-1])
            encoder = tf.layers.conv2d(prev_activated, filters=n_filters, kernel_size=5, strides=2, padding='same',
                                       kernel_initializer=w_init_encoder, use_bias=use_bias, name=ecnames[ii+1])
            encoder = tf.layers.batch_normalization(inputs=encoder, training=is_training, name=ecbnnames[ii+1])
            layers.append(encoder)

        decoder_spec = [
            (n_filter_start * 8, 0.5),  # decoder 1: [batch size, 1, 1, 512] => [batch size, 2, 2, 512*2]
            (n_filter_start * 8, 0.5),  # decoder 2: [batch size, 2, 2, 512*2] => [batch size, 4, 4, 512*2]
            (n_filter_start * 8, 0.5),  # decoder 3: [batch size, 4, 4, 512*2] => [batch size, 8, 8, 512*2]
            (n_filter_start * 8, 0.0),  # decoder 4: [batch size, 8, 8, 512*2] => [batch size, 16, 16, 512*2]
            (n_filter_start * 4, 0.0),  # decoder 5: [batch size, 16, 16, 512*2] => [batch size, 32, 32, 256*2]
            (n_filter_start * 2, 0.0),  # decoder 6: [batch size, 32, 32, 256*2] => [batch size, 64, 64, 128*2]
            (n_filter_start, 0.0),  # decoder 7: [batch size, 64, 64, 128*2] => [batch size, 128, 128, 64*2]
        ]

        # decoders
        num_encoder_layers = len(layers)
        for ii, (n_filters, dropout) in enumerate(decoder_spec):
            # dname = 'g_d{:d}_{:s}'.format(decoder_layer + 1, prefix)
            # dbnname = 'g_dbn{:d}_{:s}'.format(decoder_layer + 1, prefix)
            prev_activated = tf.nn.relu(layers[-1])
            decoder = tf.layers.conv2d_transpose(inputs=prev_activated, filters=n_filters, kernel_size=5, strides=2,
                                                 padding='same', kernel_initializer=w_init_decoder, use_bias=use_bias,
                                                 name=dcnames[ii])
            decoder = tf.layers.batch_normalization(inputs=decoder, training=is_training, name=dcbnnames[ii])

            # handle dropout (use dropout at training & inference)
            if dropout > 0.0:
                # decoder = tf.layers.dropout(decoder, rate=dropout)
                decoder = tf.layers.dropout(decoder, rate=dropout, training=True, name=dcdnames[ii])

            # handle skip layer
            skip_layer_index = num_encoder_layers - ii - 2
            concat = tf.concat([decoder, layers[skip_layer_index]], axis=3)
            layers.append(concat)

        decoder7 = tf.nn.relu(layers[-1])

        # make [batch size, 256, 256, out_channel]
        decoder8 = tf.layers.conv2d_transpose(inputs=decoder7, filters=out_channel, kernel_size=5, strides=2,
                                              padding='same', kernel_initializer=w_init_decoder, use_bias=True,
                                              name=dcnames[-1])
        out = tf.tanh(decoder8)
        return out

def discriminator(postfix, inputs, n_filter_start, alpha=0.2, stddev=0.02, reuse=False, is_training=True):
    scope_name = 'discriminator-{:s}'.format(postfix)
    with tf.variable_scope(scope_name, reuse=reuse):
        w_init = tf.truncated_normal_initializer(stddev=stddev)
        use_bias = True

        # define names for each tensor operations
        cnames = []     # convolution
        cbnnames = []   # batch-normalization
        for ii in range(1, 6):
            cname = 'd_c{:d}_{:s}'.format(ii, postfix)
            cbnname = 'd_cbn{:d}_{:s}'.format(ii, postfix)
            cnames.append(cname)
            cbnnames.append(cbnname)

        # expected inputs shape: [batch size, 256, 256, input_channel]

        # layer_1: [batch, 256, 256, input_channel] => [batch, 128, 128, 64], without batchnorm
        l1 = tf.layers.conv2d(inputs, filters=n_filter_start, kernel_size=5, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=use_bias, name=cnames[0])
        l1 = tf.maximum(alpha * l1, l1)

        # layer_2: [batch, 128, 128, 64] => [batch, 64, 64, 128], with batchnorm
        l2 = tf.layers.conv2d(l1, filters=n_filter_start * 2, kernel_size=5, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=use_bias, name=cnames[1])
        l2 = tf.layers.batch_normalization(inputs=l2, training=is_training, name=cbnnames[1])
        l2 = tf.maximum(alpha * l2, l2)

        # layer_3: [batch, 64, 64, 128] => [batch, 32, 32, 256], with batchnorm
        l3 = tf.layers.conv2d(l2, filters=n_filter_start * 4, kernel_size=5, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=use_bias, name=cnames[2])
        l3 = tf.layers.batch_normalization(inputs=l3, training=is_training, name=cbnnames[2])
        l3 = tf.maximum(alpha * l3, l3)

        # layer_4: [batch, 32, 32, 256] => [batch, 32, 32, 512], with batchnorm
        l4 = tf.layers.conv2d(l3, filters=n_filter_start * 8, kernel_size=5, strides=1, padding='same',
                              kernel_initializer=w_init, use_bias=use_bias, name=cnames[3])
        l4 = tf.layers.batch_normalization(inputs=l4, training=is_training, name=cbnnames[3])
        l4 = tf.maximum(alpha * l4, l4)

        logits = tf.layers.conv2d(l4, filters=1, kernel_size=5, strides=1, padding='same',
                                  kernel_initializer=w_init, use_bias=use_bias, name=cnames[4])
        return logits

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

    return d_loss, g_loss

def model_opt(d_loss, g_loss, g_u2v_post_fix, g_v2u_post_fix, d_u_post_fix, d_v_post_fix, learning_rate):
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
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
        self.g_model_u2v = generator(postfix='u2v', inputs=self.input_u, out_channel=self.channel_v,
                                     n_filter_start=self.n_g_filter_start, alpha=self.alpha, stddev=self.stddev,
                                     reuse=False, is_training=True)
        self.g_model_v2u = generator(postfix='v2u', inputs=self.input_v, out_channel=self.channel_u,
                                     n_filter_start=self.n_g_filter_start, alpha=self.alpha, stddev=self.stddev,
                                     reuse=False, is_training=True)

        self.g_model_u2v2u = generator(postfix='v2u', inputs=self.g_model_u2v, out_channel=self.channel_u,
                                       n_filter_start=self.n_g_filter_start, alpha=self.alpha, stddev=self.stddev,
                                       reuse=True, is_training=True)
        self.g_model_v2u2v = generator(postfix='u2v', inputs=self.g_model_v2u, out_channel=self.channel_v,
                                       n_filter_start=self.n_g_filter_start, alpha=self.alpha, stddev=self.stddev,
                                       reuse=True, is_training=True)

        # discriminators
        self.d_model_u_real_logits = discriminator(postfix='u', inputs=self.input_v,
                                                   n_filter_start=self.n_d_filter_start, alpha=self.alpha,
                                                   stddev=self.stddev, reuse=False, is_training=True)
        self.d_model_u_fake_logits = discriminator(postfix='u', inputs=self.g_model_u2v,
                                                   n_filter_start=self.n_d_filter_start, alpha=self.alpha,
                                                   stddev=self.stddev, reuse=True, is_training=True)

        self.d_model_v_real_logits = discriminator(postfix='v', inputs=self.input_u,
                                                   n_filter_start=self.n_d_filter_start, alpha=self.alpha,
                                                   stddev=self.stddev, reuse=False, is_training=True)
        self.d_model_v_fake_logits = discriminator(postfix='v', inputs=self.g_model_v2u,
                                                   n_filter_start=self.n_d_filter_start, alpha=self.alpha,
                                                   stddev=self.stddev, reuse=True, is_training=True)

        # define loss & optimizer
        self.d_loss, self.g_loss = model_loss(self.input_u, self.input_v,
                                              self.g_model_u2v2u, self.g_model_v2u2v,
                                              self.d_model_u_fake_logits, self.d_model_u_real_logits,
                                              self.d_model_v_fake_logits, self.d_model_v_real_logits,
                                              self.lambda_u, self.lambda_v)
        self.d_train_opt, self.g_train_opt = model_opt(self.d_loss, self.g_loss,
                                                       'u2v', 'v2u', 'u', 'v',
                                                       self.learning_rate)

def train(net, dataset_name, data_loader, epochs, batch_size, print_every=30):
    losses = []
    steps = 0

    # prepare saver for saving trained model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            # shuffle data randomly at every epoch
            data_loader.reset()

            for ii in range(data_loader.n_images // batch_size):
                steps += 1

                batch_image_u, batch_image_v = data_loader.get_next_batch(batch_size)

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

            # save generated images on every epochs
            for ii in range(3):
                test_image_u, test_image_v = data_loader.get_image_by_index(ii)
                g_image_u2v, g_image_u2v2u = sess.run([net.g_model_u2v, net.g_model_u2v2u],
                                                      feed_dict={net.input_u: test_image_u})
                g_image_v2u, g_image_v2u2v = sess.run([net.g_model_v2u, net.g_model_v2u2v],
                                                      feed_dict={net.input_v: test_image_v})

                image_fn = './assets/{:s}/epoch_{:d}_{:d}_tf.png'.format(dataset_name, e, ii)
                helper.save_result(image_fn,
                                   test_image_u, g_image_u2v, g_image_u2v2u,
                                   test_image_v, g_image_v2u, g_image_v2u2v)

        ckpt_fn = './checkpoints/DualGAN-{:s}.ckpt'.format(dataset_name)
        saver.save(sess, ckpt_fn)

    return losses

def test(net, dataset_name, data_loader):
    ckpt_fn = './checkpoints/DualGAN-{:s}.ckpt'.format(dataset_name)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_fn)

        for ii in range(data_loader.n_images):
            test_image_u, test_image_v = data_loader.get_image_by_index(ii)

            g_image_u2v, g_image_u2v2u = sess.run([net.g_model_u2v, net.g_model_u2v2u],
                                                  feed_dict={net.input_u: test_image_u})
            g_image_v2u, g_image_v2u2v = sess.run([net.g_model_v2u, net.g_model_v2u2v],
                                                  feed_dict={net.input_v: test_image_v})

            image_fn = './assets/{:s}/{:s}_result_{:04d}_tf.png'.format(dataset_name, dataset_name, ii)
            helper.save_result(image_fn,
                               test_image_u, g_image_u2v, g_image_u2v2u,
                               test_image_v, g_image_v2u, g_image_v2u2v)

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
        dataset_dir_u = param['dataset_dir_u']
        dataset_dir_v = param['dataset_dir_v']
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

        # prepare network
        net = DualGAN(im_size=im_size, im_channel_u=im_channel, im_channel_v=im_channel)

        if not is_test:
            # load train datasets
            train_data_loader = helper.Dataset(dataset_dir_u, dataset_dir_v, fn_ext,
                                               im_size, im_channel, im_channel, do_flip=do_flip, do_shuffle=True)

            # start training
            start_time = time.time()
            losses = train(net, dataset_name, train_data_loader, epochs, batch_size)
            end_time = time.time()
            total_time = end_time - start_time
            test_result_str = '[Training]: Data: {:s}, Epochs: {:3f}, Batch_size: {:2d}, Elapsed time: {:3f}\n'.format(
                dataset_name, epochs, batch_size, total_time)
            print(test_result_str)

            with open('./assets/test_summary.txt', 'a') as f:
                f.write(test_result_str)

        else:
            # load train datasets
            val_data_loader = helper.Dataset(dataset_dir_u, dataset_dir_v, fn_ext,
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

