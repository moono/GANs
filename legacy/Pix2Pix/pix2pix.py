# from: https://github.com/affinelayer/pix2pix-tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import helper

def model_inputs(input_height, input_width, input_channel):
    gen_inputs = tf.placeholder(tf.float32, [None, input_height, input_width, input_channel], name='gen_inputs')
    dis_inputs = tf.placeholder(tf.float32, [None, input_height, input_width, input_channel], name='dis_inputs')
    dis_targets = tf.placeholder(tf.float32, [None, input_height, input_width, input_channel], name='dis_targets')

    return gen_inputs, dis_inputs, dis_targets

def generator(inputs, out_channels, n_first_layer_filter=64, alpha=0.2, reuse=False, is_training=True):
    with tf.variable_scope('generator', reuse=reuse):
        # for convolution weights initializer
        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

        # prepare to stack layers
        layers = []

        # encoders
        # encoder_1: [batch, 256, 256, 3] => [batch, 128, 128, 64]
        layer1 = tf.layers.conv2d(inputs, filters=n_first_layer_filter, kernel_size=4, strides=2, padding='same',
                                  kernel_initializer=w_init, use_bias=False)
        layers.append(layer1)

        layer_specs = [
            n_first_layer_filter * 2,  # encoder_2: [batch, 128, 128, 64] => [batch, 64, 64, 128]
            n_first_layer_filter * 4,  # encoder_3: [batch, 64, 64, 128] => [batch, 32, 32, 256]
            n_first_layer_filter * 8,  # encoder_4: [batch, 32, 32, 256] => [batch, 16, 16, 512]
            n_first_layer_filter * 8,  # encoder_5: [batch, 16, 16, 512] => [batch, 8, 8, 512]
            n_first_layer_filter * 8,  # encoder_6: [batch, 8, 8, 512] => [batch, 4, 4, 512]
            n_first_layer_filter * 8,  # encoder_7: [batch, 4, 4, 512] => [batch, 2, 2, 512]
            n_first_layer_filter * 8,  # encoder_8: [batch, 2, 2, 512] => [batch, 1, 1, 512]
        ]
        for n_filters in layer_specs:
            layer = tf.maximum(alpha * layers[-1], layers[-1])
            layer = tf.layers.conv2d(layer, filters=n_filters, kernel_size=4, strides=2, padding='same',
                                     kernel_initializer=w_init, use_bias=False)
            layer = tf.layers.batch_normalization(inputs=layer, training=is_training)
            layers.append(layer)

        # decoders
        num_encoder_layers = len(layers)
        layer_specs = [
            (n_first_layer_filter * 8, 0.5),  # decoder_8: [batch, 1, 1, 512] => [batch, 2, 2, 512]
            (n_first_layer_filter * 8, 0.5),  # decoder_7: [batch, 2, 2, 512] => [batch, 4, 4, 512]
            (n_first_layer_filter * 8, 0.5),  # decoder_6: [batch, 4, 4, 512] => [batch, 8, 8, 512]
            (n_first_layer_filter * 8, 0.0),  # decoder_5: [batch, 8, 8, 512] => [batch, 16, 16, 512]
            (n_first_layer_filter * 4, 0.0),  # decoder_4: [batch, 16, 16, 512] => [batch, 32, 32, 256]
            (n_first_layer_filter * 2, 0.0),  # decoder_3: [batch, 32, 32, 256] => [batch, 64, 64, 128]
            (n_first_layer_filter, 0.0),  # decoder_2: [batch, 64, 64, 128] => [batch, 128, 128, 64]
        ]
        for decoder_layer, (n_filters, dropout) in enumerate(layer_specs):
            # handle skip layer
            skip_layer = num_encoder_layers - decoder_layer - 1
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                inputs = layers[-1]
            else:
                inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)
            # layer = tf.maximum(alpha * inputs, inputs)
            layer = tf.nn.relu(inputs)
            layer = tf.layers.conv2d_transpose(inputs=layer, filters=n_filters, kernel_size=4, strides=2, padding='same')
            layer = tf.layers.batch_normalization(inputs=layer, training=is_training)

            # handle dropout
            if dropout > 0.0:
               # layer = tf.layers.dropout(layer, rate=dropout)
               layer = tf.layers.dropout(layer, rate=dropout, training=True)

            # stack
            layers.append(layer)

        # decoder_1: [batch, 128, 128, 64] => [batch, 256, 256, out_channels]
        # last_layer = tf.maximum(alpha * layers[-1], layers[-1])
        last_layer = tf.nn.relu(layers[-1])
        last_layer = tf.layers.conv2d_transpose(inputs=last_layer, filters=out_channels, kernel_size=4, strides=2,
                                                padding='same')
        out = tf.tanh(last_layer)
        layers.append(last_layer)

        return out

def discriminator(inputs, targets, n_first_layer_filter=64, alpha=0.2, reuse=False, is_training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        # for convolution weights initializer
        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

        # concatenate inputs
        # [batch, 256, 256, 3] + [batch, 256, 256, 3] => [batch, 256, 256, 6]
        concat_inputs = tf.concat(values=[inputs, targets], axis=3)

        # layer_1: [batch, 256, 256, 6] => [batch, 128, 128, 64], without batchnorm
        l1 = tf.layers.conv2d(concat_inputs, filters=n_first_layer_filter, kernel_size=4, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=False)
        l1 = tf.maximum(alpha * l1, l1)

        # layer_2: [batch, 128, 128, 64] => [batch, 64, 64, 128], with batchnorm
        n_filter = n_first_layer_filter * 2
        l2 = tf.layers.conv2d(l1, filters=n_filter, kernel_size=4, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=False)
        l2 = tf.layers.batch_normalization(inputs=l2, training=is_training)
        l2 = tf.maximum(alpha * l2, l2)

        # layer_3: [batch, 64, 64, 128] => [batch, 32, 32, 256], with batchnorm
        n_filter = n_first_layer_filter * 4
        l3 = tf.layers.conv2d(l2, filters=n_filter, kernel_size=4, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=False)
        l3 = tf.layers.batch_normalization(inputs=l3, training=is_training)
        l3 = tf.maximum(alpha * l3, l3)

        # option 1. try to make same as possible in paper(same channel same receptive field size)
        # layer_4: [batch, 32, 32, 256] => [batch, 31, 31, 512], with batchnorm
        filter_4 = tf.get_variable('filter_4', [4, 4, n_first_layer_filter * 4, n_first_layer_filter * 8],
                                   dtype=tf.float32, initializer=w_init)
        padding_4 = tf.pad(l3, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        l4 = tf.nn.conv2d(padding_4, filter_4, [1, 1, 1, 1], padding='VALID')
        l4 = tf.layers.batch_normalization(inputs=l4, training=is_training)
        l4 = tf.maximum(alpha * l4, l4)

        # layer_5: [batch, 31, 31, 512] => [batch, 30, 30, 1], without batchnorm
        filter_5 = tf.get_variable('filter_5', [4, 4, n_first_layer_filter * 8, 1],
                                   dtype=tf.float32, initializer=w_init)
        padding_5 = tf.pad(l4, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        logits = tf.nn.conv2d(padding_5, filter_5, [1, 1, 1, 1], padding='VALID')
        out = tf.sigmoid(logits)

        # # option 2. try to follow tf.layer.conv2d() only match receptive field size
        # # layer_4: [batch, 32, 32, 256] => [batch, 29, 29, 512], with batchnorm
        # n_filter = n_first_layer_filter * 8
        # l4 = tf.layers.conv2d(l3, filters=n_filter, kernel_size=4, strides=1, padding='valid',
        #                       kernel_initializer=w_init, use_bias=False)
        #
        # # layer_5: [batch, 29, 29, 512] => [batch, 26, 26, 1], without batchnorm
        # n_filter = 1
        # logits = tf.layers.conv2d(l4, filters=n_filter, kernel_size=4, strides=1, padding='valid',
        #                           kernel_initializer=w_init, use_bias=False)
        # out = tf.sigmoid(logits)

        return out, logits

def model_loss(gen_inputs, dis_inputs, dis_targets, out_channels, gan_weight=1.0, l1_weight=100.0):
    # get each model outputs
    g_model = generator(gen_inputs, out_channels, reuse=False, is_training=True)
    d_model_real, d_logits_real = discriminator(dis_inputs, dis_targets, reuse=False, is_training=True)
    d_model_fake, d_logits_fake = discriminator(dis_inputs, g_model, reuse=True, is_training=True)

    # compute losses

    # discriminator loss
    # d_loss = tf.reduce_mean(-(tf.log(predict_real + eps) + tf.log(1 - predict_fake + eps)))
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    d_loss = d_loss_real + d_loss_fake

    # generator loss
    # gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + eps))
    # gen_loss_L1 = tf.reduce_mean(tf.abs(dis_targets - gen_output))
    gen_loss_gan = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))
    gen_loss_l1 = tf.reduce_mean(tf.abs(dis_targets - g_model))
    g_loss = gen_loss_gan * gan_weight + gen_loss_l1 * l1_weight

    return d_loss, gen_loss_gan, gen_loss_l1, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


def train(net, epochs, batch_size, train_input_image_dir, test_image, direction, dataset_name, print_every=30):
    losses = []
    steps = 0

    # prepare saver for saving trained model
    saver = tf.train.Saver()

    # prepare dataset
    train_dataset = helper.Dataset(train_input_image_dir, convert_to_lab_color=False, direction=direction, is_test=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for ii in range(train_dataset.n_images//batch_size):
                steps += 1

                # will return list of tuples [ (inputs, targets), (inputs, targets), ... , (inputs, targets)]
                batch_images_tuple = train_dataset.get_next_batch(batch_size)

                a = [x for x, y in batch_images_tuple]
                b = [y for x, y in batch_images_tuple]
                a = np.array(a)
                b = np.array(b)

                fd = {
                    net.dis_inputs: a,
                    net.dis_targets: b,
                    net.gen_inputs: a
                }
                d_opt_out = sess.run(net.d_train_opt, feed_dict=fd)
                g_opt_out = sess.run(net.g_train_opt, feed_dict=fd)

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = net.d_loss.eval(fd)
                    train_loss_g = net.g_loss.eval(fd)
                    train_loss_gan = net.gen_loss_GAN.eval(fd)
                    train_loss_l1 = net.gen_loss_L1.eval(fd)

                    print("Epoch {}/{}...".format(e + 1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss GAN: {:.4f}".format(train_loss_gan),
                          "Generator Loss L1: {:.4f}".format(train_loss_l1),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_gan))

            # save generated images on every epochs
            test_a = [x for x, y in test_image]
            test_a = np.array(test_a)
            gen_image = sess.run(generator(net.gen_inputs, net.input_channel, reuse=True, is_training=False),
                                 feed_dict={net.gen_inputs: test_a})
            image_fn = './assets/epoch_{:d}_tf.png'.format(e)
            helper.save_result(image_fn, gen_image)

        ckpt_fn = './checkpoints/pix2pix-{}.ckpt'.format(dataset_name)
        saver.save(sess, ckpt_fn)

    return losses

def test(net, test_input_image_dir, direction):
    # prepare dataset
    test_dataset = helper.Dataset(test_input_image_dir, convert_to_lab_color=False, direction=direction, is_test=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))

        for ii in range(test_dataset.n_images):
            test_image = test_dataset.get_image_by_index(ii)
            test_a = [x for x, y in test_image]
            test_b = [y for x, y in test_image]
            test_a = np.array(test_a)
            test_b = np.array(test_b)

            gen_image = sess.run(generator(net.gen_inputs, net.input_channel, reuse=True, is_training=True),
                                 feed_dict={net.gen_inputs: test_a})

            image_fn = './assets/test_result{:d}_tf.png'.format(ii)
            helper.save_result(image_fn, gen_image, input_image=test_a, target_image=test_b)


class Pix2Pix(object):
    def __init__(self, learning_rate):
        self.input_height, self.input_width, self.input_channel = 256, 256, 3
        self.gan_weight, self.l1_weight = 1.0, 100.0
        self.beta1 = 0.5
        self.learning_rate = learning_rate

        # build model
        tf.reset_default_graph()
        self.gen_inputs, self.dis_inputs, self.dis_targets = model_inputs(self.input_height,
                                                                          self.input_width,
                                                                          self.input_channel)
        self.d_loss, self.gen_loss_GAN, self.gen_loss_L1, self.g_loss = model_loss(self.gen_inputs,
                                                                                   self.dis_inputs,
                                                                                   self.dis_targets,
                                                                                   self.input_channel,
                                                                                   self.gan_weight,
                                                                                   self.l1_weight)
        self.d_train_opt, self.g_train_opt = model_opt(self.d_loss,
                                                       self.g_loss,
                                                       self.learning_rate,
                                                       self.beta1)

def main(do_train=True):
    assets_dir = './assets/'
    if not os.path.isdir(assets_dir):
        os.mkdir(assets_dir)

    # hyper parameters
    learning_rate = 0.0002
    n_epochs = 15
    batch_size = 4
    pix2pix = Pix2Pix(learning_rate)

    # configure parameters
    dataset_name = 'edges2handbags'
    train_name = 'train'
    test_name = 'val'
    direction = 'AtoB'

    train_input_image_dir = '../../data_set/{}/{}/'.format(dataset_name, train_name)
    test_input_image_dir = '../../data_set/{}/{}/'.format(dataset_name, test_name)

    if do_train:
        test_dataset = helper.Dataset(test_input_image_dir, convert_to_lab_color=False, direction=direction, is_test=True)
        test_single_image = test_dataset.get_image_by_index(0)

        start_time = time.time()
        losses = train(pix2pix, n_epochs, batch_size, train_input_image_dir, test_single_image, direction, dataset_name)
        end_time = time.time()
        total_time = end_time - start_time
        print('Elapsed time: ', total_time)

        fig, ax = plt.subplots()
        losses = np.array(losses)
        plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
        plt.plot(losses.T[1], label='Generator', alpha=0.5)
        plt.title("Training Losses")
        plt.legend()
        plt.savefig('./assets/losses_tf.png')
    else:
        test(pix2pix, test_input_image_dir, direction)

    return 0

if __name__ == '__main__':
    main(do_train=True)
    main(do_train=False)



