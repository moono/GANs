import tensorflow as tf
import numpy as np
import os
import time
import json

import helper

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
        self.decay = 0.9

        # Build model
        self.input_u = tf.placeholder(tf.float32, [None, im_size, im_size, im_channel_u], name='input_u')
        self.input_v = tf.placeholder(tf.float32, [None, im_size, im_size, im_channel_v], name='input_v')

        # build graph
        self.gen_A_postfix = 'gen_A'
        self.gen_B_postfix = 'gen_B'
        self.dis_A_postfix = 'dis_A'
        self.dis_B_postfix = 'dis_B'

        # generator outputs
        self.gen_A_out = self.generator(self.gen_A_postfix, self.input_u, self.channel_v, reuse=False)
        self.gen_B_out = self.generator(self.gen_B_postfix, self.input_v, self.channel_u, reuse=False)
        self.gen_AB_out = self.generator(self.gen_B_postfix, self.gen_A_out, self.channel_u, reuse=True)
        self.gen_BA_out = self.generator(self.gen_A_postfix, self.gen_B_out, self.channel_v, reuse=True)

        # discriminator outputs
        self.dis_A_real_logits = self.discriminator(self.dis_A_postfix, self.input_v, reuse=False)
        self.dis_A_fake_logits = self.discriminator(self.dis_A_postfix, self.gen_A_out, reuse=True)
        self.dis_B_real_logits = self.discriminator(self.dis_B_postfix, self.input_u, reuse=False)
        self.dis_B_fake_logits = self.discriminator(self.dis_B_postfix, self.gen_B_out, reuse=True)

        # losses
        self.gen_A_l1_loss, self.gen_B_l1_loss, self.d_loss, self.g_loss = \
            self.model_loss(self.input_u, self.input_v, self.gen_AB_out, self.gen_BA_out,
                            self.dis_A_real_logits, self.dis_A_fake_logits,
                            self.dis_B_real_logits, self.dis_B_fake_logits)

        # Optimizers
        self.d_vars, self.g_vars, self.d_train_opt, self.g_train_opt = self.model_opt(self.d_loss, self.g_loss)


    def generator(self, scope_name, inputs, out_channel, reuse=False, is_training=True):
        variable_scope_name = 'generator_{:s}'.format(scope_name)
        with tf.variable_scope(variable_scope_name, reuse=reuse):
            w_init_encoder = tf.truncated_normal_initializer(stddev=self.stddev)
            w_init_decoder = tf.random_normal_initializer(stddev=self.stddev)
            use_bias = True

            # prepare to stack layers to follow U-Net shape
            # inputs -> e1 -> e2 -> e3 -> e4 -> e5 -> e6 -> e7
            #           |     |     |     |     |     |     |  \
            #           |     |     |     |     |     |     |   e8
            #           V     V     V     V     V     V     V  /
            #     d8 <- d7 <- d6 <- d5 <- d4 <- d3 <- d2 <- d1
            layers = []

            # expected inputs shape: [batch size, 256, 256, input_channel]

            # encoders
            # make [batch size, 128, 128, 64]
            encoder1 = tf.layers.conv2d(inputs, filters=self.n_g_filter_start, kernel_size=5, strides=2, padding='same',
                                        kernel_initializer=w_init_encoder, use_bias=use_bias)
            layers.append(encoder1)

            encoder_spec = [
                self.n_g_filter_start * 2,  # encoder 2: [batch size, 128, 128, 64] => [batch size, 64, 64, 128]
                self.n_g_filter_start * 4,  # encoder 3: [batch size, 64, 64, 128] => [batch size, 32, 32, 256]
                self.n_g_filter_start * 8,  # encoder 4: [batch size, 32, 32, 256] => [batch size, 16, 16, 512]
                self.n_g_filter_start * 8,  # encoder 5: [batch size, 16, 16, 512] => [batch size, 8, 8, 512]
                self.n_g_filter_start * 8,  # encoder 6: [batch size, 8, 8, 512] => [batch size, 4, 4, 512]
                self.n_g_filter_start * 8,  # encoder 7: [batch size, 4, 4, 512] => [batch size, 2, 2, 512]
                self.n_g_filter_start * 8,  # encoder 8: [batch size, 2, 2, 512] => [batch size, 1, 1, 512]
            ]

            for ii, n_filters in enumerate(encoder_spec):
                prev_activated = tf.maximum(self.alpha * layers[-1], layers[-1])
                encoder = tf.layers.conv2d(prev_activated, filters=n_filters, kernel_size=5, strides=2, padding='same',
                                           kernel_initializer=w_init_encoder, use_bias=use_bias)
                encoder = tf.layers.batch_normalization(inputs=encoder, training=is_training)
                layers.append(encoder)

            decoder_spec = [
                (self.n_g_filter_start * 8, 0.5),  # decoder 1: [batch size, 1, 1, 512] => [batch size, 2, 2, 512*2]
                (self.n_g_filter_start * 8, 0.5),  # decoder 2: [batch size, 2, 2, 512*2] => [batch size, 4, 4, 512*2]
                (self.n_g_filter_start * 8, 0.5),  # decoder 3: [batch size, 4, 4, 512*2] => [batch size, 8, 8, 512*2]
                (self.n_g_filter_start * 8, 0.0),  # decoder 4: [batch size, 8, 8, 512*2] => [batch size, 16, 16, 512*2]
                (self.n_g_filter_start * 4, 0.0),  # decoder 5: [batch size, 16, 16, 512*2] => [batch size, 32, 32, 256*2]
                (self.n_g_filter_start * 2, 0.0),  # decoder 6: [batch size, 32, 32, 256*2] => [batch size, 64, 64, 128*2]
                (self.n_g_filter_start, 0.0),  # decoder 7: [batch size, 64, 64, 128*2] => [batch size, 128, 128, 64*2]
            ]

            # decoders
            num_encoder_layers = len(layers)
            for decoder_layer, (n_filters, dropout_rate) in enumerate(decoder_spec):
                prev_activated = tf.nn.relu(layers[-1])
                decoder = tf.layers.conv2d_transpose(inputs=prev_activated, filters=n_filters, kernel_size=5, strides=2,
                                                     padding='same', kernel_initializer=w_init_decoder,
                                                     use_bias=use_bias)
                decoder = tf.layers.batch_normalization(inputs=decoder, training=is_training)

                # handle dropout (use dropout at training & inference)
                if dropout_rate > 0.0:
                    # decoder = tf.layers.dropout(decoder, rate=dropout_rate)
                    decoder = tf.layers.dropout(decoder, rate=dropout_rate, training=True)

                # handle skip layer
                skip_layer_index = num_encoder_layers - decoder_layer - 2
                concat = tf.concat([decoder, layers[skip_layer_index]], axis=3)
                layers.append(concat)

            decoder7 = tf.nn.relu(layers[-1])

            # make [batch size, 256, 256, out_channel]
            decoder8 = tf.layers.conv2d_transpose(inputs=decoder7, filters=out_channel, kernel_size=5, strides=2,
                                                  padding='same', kernel_initializer=w_init_decoder,
                                                  use_bias=use_bias)
            out = tf.tanh(decoder8)
            return out

    def discriminator(self, scope_name, inputs, reuse=False, is_training=True):
        variable_scope_name = 'discriminator_{:s}'.format(scope_name)
        with tf.variable_scope(variable_scope_name, reuse=reuse):
            w_init = tf.truncated_normal_initializer(stddev=self.stddev)
            use_bias = True

            # expected inputs shape: [batch size, 256, 256, input_channel]
            # layer_1: [batch, 256, 256, input_channel] => [batch, 128, 128, 64], without batchnorm
            l1 = tf.layers.conv2d(inputs, filters=self.n_d_filter_start, kernel_size=5, strides=2, padding='same',
                                  kernel_initializer=w_init, use_bias=use_bias)
            l1 = tf.maximum(self.alpha * l1, l1)

            # layer_2: [batch, 128, 128, 64] => [batch, 64, 64, 128], with batchnorm
            l2 = tf.layers.conv2d(l1, filters=self.n_d_filter_start * 2, kernel_size=5, strides=2, padding='same',
                                  kernel_initializer=w_init, use_bias=use_bias)
            l2 = tf.layers.batch_normalization(inputs=l2, training=is_training)
            l2 = tf.maximum(self.alpha * l2, l2)

            # layer_3: [batch, 64, 64, 128] => [batch, 32, 32, 256], with batchnorm
            l3 = tf.layers.conv2d(l2, filters=self.n_d_filter_start * 4, kernel_size=5, strides=2, padding='same',
                                  kernel_initializer=w_init, use_bias=use_bias)
            l3 = tf.layers.batch_normalization(inputs=l3, training=is_training)
            l3 = tf.maximum(self.alpha * l3, l3)

            # layer_4: [batch, 32, 32, 256] => [batch, 32, 32, 512], with batchnorm
            l4 = tf.layers.conv2d(l3, filters=self.n_d_filter_start * 8, kernel_size=5, strides=1, padding='same',
                                  kernel_initializer=w_init, use_bias=use_bias)
            l4 = tf.layers.batch_normalization(inputs=l4, training=is_training)
            l4 = tf.maximum(self.alpha * l4, l4)

            logits = tf.layers.conv2d(l4, filters=1, kernel_size=5, strides=1, padding='same',
                                      kernel_initializer=w_init, use_bias=use_bias)
            # out = tf.sigmoid(logits)

            return logits

    def model_loss(self, input_u, input_v, gen_AB_out, gen_BA_out,
                   dis_A_real_logits, dis_A_fake_logits, dis_B_real_logits, dis_B_fake_logits):
        # shorten cross entropy loss calculation
        def celoss_ones(logits):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))

        def celoss_zeros(logits):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)))

        # discriminator losses
        dis_A_real_loss = celoss_ones(dis_A_real_logits)
        dis_A_fake_loss = celoss_zeros(dis_A_fake_logits)
        dis_B_real_loss = celoss_ones(dis_B_real_logits)
        dis_B_fake_loss = celoss_zeros(dis_B_fake_logits)
        d_loss_A = dis_A_real_loss + dis_A_fake_loss
        d_loss_B = dis_B_real_loss + dis_B_fake_loss
        d_loss = d_loss_A + d_loss_B

        # generator losses
        gen_A_loss = celoss_ones(dis_A_fake_logits)
        gen_B_loss = celoss_ones(dis_B_fake_logits)
        gen_A_l1_loss = tf.reduce_mean(tf.abs(input_u - gen_AB_out))
        gen_B_l1_loss = tf.reduce_mean(tf.abs(input_v - gen_BA_out))

        g_loss_A = gen_A_loss + self.lambda_v * gen_B_l1_loss
        g_loss_B = gen_B_loss + self.lambda_u * gen_A_l1_loss
        g_loss = g_loss_A + g_loss_B

        return gen_A_l1_loss, gen_B_l1_loss, d_loss, g_loss

    def model_opt(self, d_loss, g_loss):
        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        print(len(d_vars))
        print(len(g_vars))

        # Optimizers
        d_train_opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay).minimize(g_loss, var_list=g_vars)

        return d_vars, g_vars, d_train_opt, g_train_opt

def train(net, dataset_name, train_data_loader, val_data_loader, epochs, batch_size, print_every=30, save_every=100):
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
                    train_loss_A_l1 = net.gen_A_l1_loss.eval(fd)
                    train_loss_B_l1 = net.gen_B_l1_loss.eval(fd)

                    print("Epoch {}/{}...".format(e + 1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g),
                          "A-L1 Loss: {:.4f}".format(train_loss_A_l1),
                          "B-L1 Loss: {:.4f}".format(train_loss_B_l1))

                if steps % save_every == 0:
                    # save generated images on every epochs
                    random_index = np.random.randint(0, val_data_loader.n_images)
                    test_image_u, test_image_v = val_data_loader.get_image_by_index(random_index)
                    gen_A_out, gen_AB_out = sess.run([net.gen_A_out, net.gen_AB_out], feed_dict={net.input_u: test_image_u})
                    gen_B_out, gen_BA_out = sess.run([net.gen_B_out, net.gen_BA_out], feed_dict={net.input_v: test_image_v})

                    image_fn = './assets/{:s}/epoch_{:d}-{:d}_tf.png'.format(dataset_name, e, steps)
                    helper.save_result(image_fn,
                                       test_image_u, gen_A_out, gen_AB_out,
                                       test_image_v, gen_B_out, gen_BA_out)

        ckpt_fn = './checkpoints/DualGAN-{:s}.ckpt'.format(dataset_name)
        saver.save(sess, ckpt_fn)

    return

def test(net, dataset_name, val_data_loader):
    ckpt_fn = './checkpoints/DualGAN-{:s}.ckpt'.format(dataset_name)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_fn)

        for ii in range(val_data_loader.n_images):
            test_image_u, test_image_v = val_data_loader.get_image_by_index(ii)

            gen_A_out, gen_AB_out = sess.run([net.gen_A_out, net.gen_AB_out], feed_dict={net.input_u: test_image_u})
            gen_B_out, gen_BA_out = sess.run([net.gen_B_out, net.gen_BA_out], feed_dict={net.input_v: test_image_v})

            image_fn = './assets/{:s}/{:s}_result_{:04d}_tf.png'.format(dataset_name, dataset_name, ii)
            helper.save_result(image_fn,
                               test_image_u, gen_A_out, gen_AB_out,
                               test_image_v, gen_B_out, gen_BA_out)

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
            train(net, dataset_name, train_data_loader, val_data_loader, epochs, batch_size)
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


if __name__ == '__main__':
    main()

