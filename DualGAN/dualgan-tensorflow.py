import tensorflow as tf
import numpy as np
import os
import time


class DualGAN(object):
    def __init__(self, im_height, im_width, im_channel):
        #
        self.height, self.width, self.channel = im_height, im_width, im_channel

        self.n_g_filter_start = 64
        self.n_d_filter_start = 64
        self.alpha = 0.2
        self.stddev = 0.02

        # loss related
        self.beta1 = 0.5
        self.lambda_u = 20.0
        self.lambda_v = 20.0
        self.learning_rate = 0.0005

        # Build model
        self.input_u = tf.placeholder(tf.float32, [None, im_height, im_width, im_channel], name='input_u')
        self.input_v = tf.placeholder(tf.float32, [None, im_height, im_width, im_channel], name='input_v')

        # define loss
        self.d_loss, self.g_loss = self.model_loss(self.input_u, self.input_v)
        self.d_train_opt, self.g_train_opt = self.model_opt(self.d_loss, self.g_loss)


    def model_loss(self, input_u, input_v):
        # generators
        generator_u_to_v = self.generator('u-to-v', input_u, reuse=False, is_training=True)
        generator_v_to_u = self.generator('v-to-u', input_v, reuse=False, is_training=True)
        generator_u_to_v_to_u = self.generator('v-to-u', generator_u_to_v, reuse=True, is_training=True)
        generator_v_to_u_to_v = self.generator('u-to-v', generator_v_to_u, reuse=True, is_training=True)

        # discriminators
        discriminator_u_fake, discriminator_u_fake_logits = self.discriminator('u', generator_u_to_v,
                                                                               reuse=False, is_training=True)
        discriminator_u_real, discriminator_u_real_logits = self.discriminator('u', input_v,
                                                                               reuse=True, is_training=True)
        discriminator_v_fake, discriminator_v_fake_logits = self.discriminator('v', generator_v_to_u,
                                                                               reuse=False, is_training=True)
        discriminator_v_real, discriminator_v_real_logits = self.discriminator('v', input_u,
                                                                               reuse=True, is_training=True)

        # loss
        # discriminator losses
        d_loss_real_u = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_u_real_logits,
                                                    labels=tf.ones_like(discriminator_u_real)))
        d_loss_fake_u = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_u_fake_logits,
                                                    labels=tf.zeros_like(discriminator_u_fake)))
        d_loss_u = d_loss_real_u + d_loss_fake_u

        d_loss_real_v = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_v_real_logits,
                                                    labels=tf.ones_like(discriminator_v_real)))
        d_loss_fake_v = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_v_fake_logits,
                                                    labels=tf.zeros_like(discriminator_v_fake)))
        d_loss_v = d_loss_real_v + d_loss_fake_v
        d_loss = d_loss_u + d_loss_v

        # generator losses
        g_loss_l1_u = tf.reduce_mean(tf.abs(generator_u_to_v_to_u - input_u))
        g_loss_l1_v = tf.reduce_mean(tf.abs(generator_v_to_u_to_v - input_v))

        g_loss_gan_u = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_u_fake_logits,
                                                    labels=tf.ones_like(discriminator_u_fake)))
        g_loss_gan_v = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_v_fake_logits,
                                                    labels=tf.ones_like(discriminator_v_fake)))

        g_loss_u = g_loss_gan_u + self.lambda_v * g_loss_l1_v
        g_loss_v = g_loss_gan_v + self.lambda_u * g_loss_l1_u
        g_loss = g_loss_u + g_loss_v

        return d_loss, g_loss


    def model_opt(self, d_loss, g_loss):
        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimize
        decay = 0.9
        d_train_opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=decay).minimize(d_loss, var_list=d_vars)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            g_train_opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=decay).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt

    def generator(self, scope_name, inputs, reuse=False, is_training=True):
        variable_scope_name = 'generator_{:s}'.format(scope_name)
        with tf.variable_scope(variable_scope_name, reuse=reuse):
            w_init_encoder = tf.truncated_normal_initializer(stddev=self.stddev)
            w_init_decoder = tf.random_normal_initializer(stddev=self.stddev)

            # prepare to stack layers to follow U-Net shape
            # inputs -> e1 -> e2 -> e3 -> e4 -> e5 -> e6 -> e7
            #           |     |     |     |     |     |     |  \
            #           |     |     |     |     |     |     |   e8
            #           V     V     V     V     V     V     V  /
            #     d8 <- d7 <- d6 <- d5 <- d4 <- d3 <- d2 <- d1
            layers = []

            # expected inputs shape: [batch size, 256, 256, 3]

            # encoders
            # make [batch size, 128, 128, 64]
            encoder1 = tf.layers.conv2d(inputs, filters=self.n_g_filter_start, kernel_size=5, strides=2,
                                        padding='same', kernel_initializer=w_init_encoder, use_bias=True)
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

            for n_filters in encoder_spec:
                prev_activated = tf.maximum(self.alpha * layers[-1], layers[-1])
                encoder = tf.layers.conv2d(prev_activated, filters=n_filters, kernel_size=5, strides=2, padding='same',
                                           kernel_initializer=w_init_encoder, use_bias=True)
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
            for decoder_layer, (n_filters, dropout) in enumerate(decoder_spec):
                prev_activated = tf.nn.relu(layers[-1])
                decoder = tf.layers.conv2d_transpose(inputs=prev_activated, filters=n_filters, kernel_size=5, strides=2,
                                                     padding='same', kernel_initializer=w_init_decoder, use_bias=True)
                decoder = tf.layers.batch_normalization(inputs=decoder, training=is_training)

                # handle dropout
                if dropout > 0.0:
                    decoder = tf.layers.dropout(decoder, rate=dropout)

                # handle skip layer
                skip_layer_index = num_encoder_layers - decoder_layer - 2
                concat = tf.concat([decoder, layers[skip_layer_index]], axis=3)
                layers.append(concat)

            decoder7 = tf.nn.relu(layers[-1])

            # make [batch size, 256, 256, 3]
            decoder8 = tf.layers.conv2d_transpose(inputs=decoder7, filters=self.channel, kernel_size=5, strides=2,
                                                 padding='same', kernel_initializer=w_init_decoder, use_bias=True)
            out = tf.tanh(decoder8)
            return out

    def discriminator(self, scope_name, inputs, reuse=False, is_training=True):
        variable_scope_name = 'discriminator_{:s}'.format(scope_name)
        with tf.variable_scope(variable_scope_name, reuse=reuse):
            w_init = tf.truncated_normal_initializer(stddev=self.stddev)

            # expected inputs shape: [batch size, 256, 256, 3]

            # layer_1: [batch, 256, 256, 3] => [batch, 128, 128, 64], without batchnorm
            l1 = tf.layers.conv2d(inputs, filters=self.n_d_filter_start, kernel_size=5, strides=2, padding='same',
                                  kernel_initializer=w_init, use_bias=True)
            l1 = tf.maximum(self.alpha * l1, l1)

            # layer_2: [batch, 128, 128, 64] => [batch, 64, 64, 128], with batchnorm
            l2 = tf.layers.conv2d(l1, filters=self.n_d_filter_start * 2, kernel_size=5, strides=2, padding='same',
                                  kernel_initializer=w_init, use_bias=True)
            l2 = tf.layers.batch_normalization(inputs=l2, training=is_training)
            l2 = tf.maximum(self.alpha * l2, l2)

            # layer_3: [batch, 64, 64, 128] => [batch, 32, 32, 256], with batchnorm
            l3 = tf.layers.conv2d(l2, filters=self.n_d_filter_start * 4, kernel_size=5, strides=2, padding='same',
                                  kernel_initializer=w_init, use_bias=True)
            l3 = tf.layers.batch_normalization(inputs=l3, training=is_training)
            l3 = tf.maximum(self.alpha * l3, l3)

            # layer_4: [batch, 32, 32, 256] => [batch, 32, 32, 512], with batchnorm
            l4 = tf.layers.conv2d(l3, filters=self.n_d_filter_start * 8, kernel_size=5, strides=1, padding='same',
                                  kernel_initializer=w_init, use_bias=True)
            l4 = tf.layers.batch_normalization(inputs=l4, training=is_training)
            l4 = tf.maximum(self.alpha * l4, l4)

            logits = tf.layers.conv2d(l4, filters=1, kernel_size=5, strides=1, padding='same',
                                      kernel_initializer=w_init, use_bias=True)
            out = tf.sigmoid(logits)

            return out, logits



def main():
    print()

if __name__ == '__main__':
    main()
