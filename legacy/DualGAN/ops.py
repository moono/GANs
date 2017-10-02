import tensorflow as tf

def generator_unet_256(inputs, n_filter_start, n_out_channel, scope_name='generator', kernel_size=5,
                       w_init_encoder=None, w_init_decoder=None, alpha=0.2, drop_rate=0.5, use_bias=False,
                       reuse=False, is_training=True):
    """
    :param inputs: input Tensor
    :param n_filter_start: size of filter in first encoder
    :param n_out_channel: last decoder out channel
    :param scope_name: scope name
    :param kernel_size: kernel size to use in encoder & decoder
    :param w_init_encoder: weight initializer for encoder
    :param w_init_decoder: weight initializer for decoder
    :param alpha: leaky ReLU alpha
    :param drop_rate: dropout layer rate
    :param use_bias: set to use bias term on conv2d & conv2d_transpose
    :param reuse: reuse entire Variables
    :param is_training: useful for tf.layers.batch_normalization
    :return: generator out
    """
    with tf.variable_scope(scope_name, reuse=reuse):
        # prepare to stack layers to follow U-Net shape
        # inputs -> e1 -> e2 -> e3 -> e4 -> e5 -> e6 -> e7
        #           |     |     |     |     |     |     |  \
        #           |     |     |     |     |     |     |   e8
        #           V     V     V     V     V     V     V  /
        #     d8 <- d7 <- d6 <- d5 <- d4 <- d3 <- d2 <- d1

        # expected inputs shape: [batch size, 256, 256, input_channel]
        # encoders
        # encoder 1: [batch size, 256, 256, input_channel] ==> [batch size, 128, 128, 64]
        e1 = tf.layers.conv2d(inputs, filters=n_filter_start, kernel_size=kernel_size, strides=2, padding='same',
                              kernel_initializer=w_init_encoder, use_bias=use_bias)

        # encoder 2: [batch size, 128, 128, 64] => [batch size, 64, 64, 128]
        e2 = tf.maximum(alpha * e1, e1)
        e2 = tf.layers.conv2d(e2, filters=n_filter_start * 2, kernel_size=kernel_size, strides=2, padding='same',
                              kernel_initializer=w_init_encoder, use_bias=use_bias)
        e2 = tf.layers.batch_normalization(inputs=e2, training=is_training)

        # encoder 3: [batch size, 64, 64, 128] => [batch size, 32, 32, 256]
        e3 = tf.maximum(alpha * e2, e2)
        e3 = tf.layers.conv2d(e3, filters=n_filter_start * 4, kernel_size=kernel_size, strides=2, padding='same',
                              kernel_initializer=w_init_encoder, use_bias=use_bias)
        e3 = tf.layers.batch_normalization(inputs=e3, training=is_training)

        # encoder 4: [batch size, 32, 32, 256] => [batch size, 16, 16, 512]
        e4 = tf.maximum(alpha * e3, e3)
        e4 = tf.layers.conv2d(e4, filters=n_filter_start * 8, kernel_size=kernel_size, strides=2, padding='same',
                              kernel_initializer=w_init_encoder, use_bias=use_bias)
        e4 = tf.layers.batch_normalization(inputs=e4, training=is_training)

        # encoder 5: [batch size, 16, 16, 512] => [batch size, 8, 8, 512]
        e5 = tf.maximum(alpha * e4, e4)
        e5 = tf.layers.conv2d(e5, filters=n_filter_start * 8, kernel_size=kernel_size, strides=2, padding='same',
                              kernel_initializer=w_init_encoder, use_bias=use_bias)
        e5 = tf.layers.batch_normalization(inputs=e5, training=is_training)

        # encoder 6: [batch size, 8, 8, 512] => [batch size, 4, 4, 512]
        e6 = tf.maximum(alpha * e5, e5)
        e6 = tf.layers.conv2d(e6, filters=n_filter_start * 8, kernel_size=kernel_size, strides=2, padding='same',
                              kernel_initializer=w_init_encoder, use_bias=use_bias)
        e6 = tf.layers.batch_normalization(inputs=e6, training=is_training)

        # encoder 7: [batch size, 4, 4, 512] => [batch size, 2, 2, 512]
        e7 = tf.maximum(alpha * e6, e6)
        e7 = tf.layers.conv2d(e7, filters=n_filter_start * 8, kernel_size=kernel_size, strides=2, padding='same',
                              kernel_initializer=w_init_encoder, use_bias=use_bias)
        e7 = tf.layers.batch_normalization(inputs=e7, training=is_training)

        # encoder 8: [batch size, 2, 2, 512] => [batch size, 1, 1, 512]
        e8 = tf.maximum(alpha * e7, e7)
        e8 = tf.layers.conv2d(e8, filters=n_filter_start * 8, kernel_size=kernel_size, strides=2, padding='same',
                              kernel_initializer=w_init_encoder, use_bias=use_bias)
        e8 = tf.layers.batch_normalization(inputs=e8, training=is_training)

        # decoders
        # decoder 1: [batch size, 1, 1, 512] => [batch size, 2, 2, 512*2] with dropout
        d1 = tf.nn.relu(e8)
        d1 = tf.layers.conv2d_transpose(inputs=d1, filters=n_filter_start * 8, kernel_size=5, strides=2, padding='same',
                                        kernel_initializer=w_init_decoder, use_bias=use_bias)
        d1 = tf.layers.batch_normalization(inputs=d1, training=is_training)
        d1 = tf.layers.dropout(d1, rate=drop_rate, training=True)
        d1 = tf.concat([d1, e7], axis=3)

        # decoder 2: [batch size, 2, 2, 512*2] => [batch size, 4, 4, 512*2] with dropout
        d2 = tf.nn.relu(d1)
        d2 = tf.layers.conv2d_transpose(inputs=d2, filters=n_filter_start * 8, kernel_size=5, strides=2, padding='same',
                                        kernel_initializer=w_init_decoder, use_bias=use_bias)
        d2 = tf.layers.batch_normalization(inputs=d2, training=is_training)
        d2 = tf.layers.dropout(d2, rate=drop_rate, training=True)
        d2 = tf.concat([d2, e6], axis=3)

        # decoder 3: [batch size, 4, 4, 512*2] => [batch size, 8, 8, 512*2] with dropout
        d3 = tf.nn.relu(d2)
        d3 = tf.layers.conv2d_transpose(inputs=d3, filters=n_filter_start * 8, kernel_size=5, strides=2, padding='same',
                                        kernel_initializer=w_init_decoder, use_bias=use_bias)
        d3 = tf.layers.batch_normalization(inputs=d3, training=is_training)
        d3 = tf.layers.dropout(d3, rate=drop_rate, training=True)
        d3 = tf.concat([d3, e5], axis=3)

        # decoder 4: [batch size, 8, 8, 512*2] => [batch size, 16, 16, 512*2]
        d4 = tf.nn.relu(d3)
        d4 = tf.layers.conv2d_transpose(inputs=d4, filters=n_filter_start * 8, kernel_size=5, strides=2, padding='same',
                                        kernel_initializer=w_init_decoder, use_bias=use_bias)
        d4 = tf.layers.batch_normalization(inputs=d4, training=is_training)
        d4 = tf.concat([d4, e4], axis=3)

        # decoder 5: [batch size, 16, 16, 512*2] => [batch size, 32, 32, 256*2]
        d5 = tf.nn.relu(d4)
        d5 = tf.layers.conv2d_transpose(inputs=d5, filters=n_filter_start * 4, kernel_size=5, strides=2, padding='same',
                                        kernel_initializer=w_init_decoder, use_bias=use_bias)
        d5 = tf.layers.batch_normalization(inputs=d5, training=is_training)
        d5 = tf.concat([d5, e3], axis=3)

        # decoder 6: [batch size, 32, 32, 256*2] => [batch size, 64, 64, 128*2]
        d6 = tf.nn.relu(d5)
        d6 = tf.layers.conv2d_transpose(inputs=d6, filters=n_filter_start * 2, kernel_size=5, strides=2, padding='same',
                                        kernel_initializer=w_init_decoder, use_bias=use_bias)
        d6 = tf.layers.batch_normalization(inputs=d6, training=is_training)
        d6 = tf.concat([d6, e2], axis=3)

        # decoder 7: [batch size, 64, 64, 128*2] => [batch size, 128, 128, 64*2]
        d7 = tf.nn.relu(d6)
        d7 = tf.layers.conv2d_transpose(inputs=d7, filters=n_filter_start, kernel_size=5, strides=2, padding='same',
                                        kernel_initializer=w_init_decoder, use_bias=use_bias)
        d7 = tf.layers.batch_normalization(inputs=d7, training=is_training)
        d7 = tf.concat([d7, e1], axis=3)

        # decoder 8: [batch size, 128, 128, 64*2] =>[batch size, 256, 256, n_out_channel]
        d8 = tf.nn.relu(d7)
        d8 = tf.layers.conv2d_transpose(inputs=d8, filters=n_out_channel, kernel_size=5, strides=2, padding='same',
                                        kernel_initializer=w_init_decoder, use_bias=use_bias)

        out = tf.tanh(d8)
        return out

def discriminator_unet_256(inputs, n_filter_start, scope_name='discriminator', kernel_size=5,
                           w_init=None, alpha=0.2, use_bias=False,
                           reuse=False, is_training=True):
    """
        :param inputs: input Tensor
        :param n_filter_start: size of filter in first encoder
        :param scope_name: scope name
        :param kernel_size: kernel size to use in encoder & decoder
        :param w_init: weight initializer for encoder
        :param alpha: leaky ReLU alpha
        :param use_bias: set to use bias term on conv2d & conv2d_transpose
        :param reuse: reuse entire Variables
        :param is_training: useful for tf.layers.batch_normalization
        :return: generator out
        """
    with tf.variable_scope(scope_name, reuse=reuse):
        # expected inputs shape: [batch size, 256, 256, input_channel]
        # layer_1: [batch, 256, 256, input_channel] => [batch, 128, 128, 64]
        l1 = tf.layers.conv2d(inputs, filters=n_filter_start, kernel_size=kernel_size, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=use_bias)
        l1 = tf.maximum(alpha * l1, l1)

        # layer_2: [batch, 128, 128, 64] => [batch, 64, 64, 128]
        l2 = tf.layers.conv2d(l1, filters=n_filter_start * 2, kernel_size=kernel_size, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=use_bias)
        l2 = tf.layers.batch_normalization(inputs=l2, training=is_training)
        l2 = tf.maximum(alpha * l2, l2)

        # layer_3: [batch, 64, 64, 128] => [batch, 32, 32, 256]
        l3 = tf.layers.conv2d(l2, filters=n_filter_start * 4, kernel_size=kernel_size, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=use_bias)
        l3 = tf.layers.batch_normalization(inputs=l3, training=is_training)
        l3 = tf.maximum(alpha * l3, l3)

        # layer_4: [batch, 32, 32, 256] => [batch, 32, 32, 512]
        l4 = tf.layers.conv2d(l3, filters=n_filter_start * 8, kernel_size=kernel_size, strides=1, padding='same',
                              kernel_initializer=w_init, use_bias=use_bias)
        l4 = tf.layers.batch_normalization(inputs=l4, training=is_training)
        l4 = tf.maximum(alpha * l4, l4)

        # layer_5: [batch, 32, 32, 512] => [batch, 32, 32, 1]
        logits = tf.layers.conv2d(l4, filters=1, kernel_size=kernel_size, strides=1, padding='same',
                                  kernel_initializer=w_init, use_bias=use_bias)
        return logits
