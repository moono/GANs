import tensorflow as tf


def encoder(e_in, n_f, n_k, w_init, is_training, repeat=1, is_first_layer=False):
    prev_in = e_in

    # apply stride 1 conv2d repeated times
    for ii in range(repeat):
        prev_in = tf.layers.conv2d(prev_in, n_f, n_k, 1, 'same', kernel_initializer=w_init)
        if ii == 0 and is_first_layer == True:
            prev_in = tf.nn.relu(prev_in)
        else:
            prev_in = tf.layers.batch_normalization(prev_in, training=is_training)
            prev_in = tf.nn.relu(prev_in)

    # apply stride 2 conv2d
    e = tf.layers.conv2d(prev_in, n_f, n_k, 2, 'same', kernel_initializer=w_init)
    e = tf.layers.batch_normalization(e, training=is_training)
    e = tf.nn.relu(e)
    return e


def decoder(d_in, n_f, n_k, w_init, is_training, drop_rate=0.0, repeat=1, is_last_layer=False):
    prev_in = d_in

    # apply stride 1 conv2d_transpose repeated times
    for ii in range(repeat):
        prev_in = tf.layers.conv2d_transpose(inputs=prev_in, filters=n_f, kernel_size=n_k, strides=1, padding='same',
                                             kernel_initializer=w_init)
        if drop_rate != 0.0:
            prev_in = tf.layers.dropout(prev_in, rate=drop_rate, training=True)
        prev_in = tf.layers.batch_normalization(prev_in, training=is_training)
        prev_in = tf.nn.relu(prev_in)

    # apply stride 2 conv2d_transpose
    d = tf.layers.conv2d_transpose(inputs=prev_in, filters=n_f, kernel_size=n_k, strides=2, padding='same',
                                   kernel_initializer=w_init)

    if is_last_layer:
        d = tf.tanh(d)
    else:
        d = tf.layers.batch_normalization(d, training=is_training)
        d = tf.layers.dropout(d, rate=drop_rate, training=True)
        d = tf.nn.relu(d)

    return d


def generator(inputs, cond=None, reuse=False, is_training=True):
    with tf.variable_scope('generator', reuse=reuse):
        # set parameters
        w_init = tf.contrib.layers.xavier_initializer()
        n_f = 64
        n_k = 4
        r_drop = 0.5
        r_nodrop = 0.0

        concated = inputs
        if cond is not None:
            concated = tf.concat([inputs, cond], axis=3)

        # encoder 1: [batch size, 512, 512, 6] ==> [batch size, 256, 256, 64]
        e1 = encoder(concated, n_f, n_k, w_init, is_training, repeat=1, is_first_layer=True)

        # encoder 2: [batch size, 256, 256, 64] ==> [batch size, 128, 128, 128]
        e2 = encoder(e1, n_f * 2, n_k, w_init, is_training, repeat=1)

        # encoder 3: [batch size, 128, 128, 128] ==> [batch size, 64, 64, 256]
        e3 = encoder(e2, n_f * 4, n_k, w_init, is_training, repeat=1)

        # encoder 4: [batch size, 64, 64, 256] ==> [batch size, 32, 32, 512]
        e4 = encoder(e3, n_f * 8, n_k, w_init, is_training, repeat=2)

        # encoder 5: [batch size, 32, 32, 512] ==> [batch size, 16, 16, 512]
        e5 = encoder(e4, n_f * 8, n_k, w_init, is_training, repeat=2)

        # encoder 6: [batch size, 16, 16, 512] ==> [batch size, 8, 8, 512]
        e6 = encoder(e5, n_f * 8, n_k, w_init, is_training, repeat=2)

        # encoder 7: [batch size, 8, 8, 512] ==> [batch size, 4, 4, 512]
        e7 = encoder(e6, n_f * 8, n_k, w_init, is_training, repeat=2)

        # encoder 8: [batch size, 4, 4, 512] ==> [batch size, 2, 2, 512]
        e8 = encoder(e7, n_f * 8, n_k, w_init, is_training, repeat=2)

        # encoder 9: [batch size, 2, 2, 512] ==> [batch size, 1, 1, 512]
        e9 = encoder(e8, n_f * 8, n_k, w_init, is_training, repeat=2)

        # decoder 0: [batch size, 1, 1, 512] ==> [batch size, 2, 2, 512]
        d0 = decoder(e9, n_f * 8, n_k, w_init, is_training, r_drop, repeat=2)

        # decoder 1: [batch size, 2, 2, 512] ==> [batch size, 4, 4, 512]
        concated_08 = tf.concat([d0, e8], axis=3)
        d1 = decoder(concated_08, n_f * 8, n_k, w_init, is_training, r_drop, repeat=2)

        # decoder 2: [batch size, 4, 4, 512 * 2] ==> [batch size, 8, 8, 512]
        concated_17 = tf.concat([d1, e7], axis=3)
        d2 = decoder(concated_17, n_f * 8, n_k, w_init, is_training, r_drop, repeat=2)

        # decoder 3: [batch size, 8, 8, 512 * 2] ==> [batch size, 16, 16, 512]
        concated_26 = tf.concat([d2, e6], axis=3)
        d3 = decoder(concated_26, n_f * 8, n_k, w_init, is_training, r_drop, repeat=2)

        # decoder 4: [batch size, 16, 16, 512 * 2] ==> [batch size, 32, 32, 512]
        concated_35 = tf.concat([d3, e5], axis=3)
        d4 = decoder(concated_35, n_f * 8, n_k, w_init, is_training, r_drop, repeat=2)

        # decoder 4: [batch size, 32, 32, 512 * 2] ==> [batch size, 64, 64, 256]
        concated_44 = tf.concat([d4, e4], axis=3)
        d5 = decoder(concated_44, n_f * 4, n_k, w_init, is_training, r_nodrop, repeat=2)

        # decoder 5: [batch size, 64, 64, 256 * 2] ==> [batch size, 128, 128, 128]
        concated_53 = tf.concat([d5, e3], axis=3)
        d6 = decoder(concated_53, n_f * 2, n_k, w_init, is_training, r_nodrop, repeat=1)

        # decoder 6: [batch size, 128, 128, 128 * 2] ==> [batch size, 256, 256, 64]
        concated_62 = tf.concat([d6, e2], axis=3)
        d7 = decoder(concated_62, n_f, n_k, w_init, is_training, r_nodrop, repeat=1)

        # decoder 13: [batch size, 256, 256, 64 * 2] ==> [batch size, 512, 512, 3]
        concated_71 = tf.concat([d7, e1], axis=3)
        d8 = decoder(concated_71, 3, n_k, w_init, is_training, r_nodrop, repeat=1, is_last_layer=True)

        return d8


def discriminator(inputs, targets, reuse=False, is_training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()
        alpha = 0.2
        n_f = 64
        n_k = 4

        # concatenate inputs
        # [batch, 512, 512, 3] + [batch, 512, 512, 3] => [batch, 512, 512, 6]
        concat_inputs = tf.concat(values=[inputs, targets], axis=3)

        # layer_1: [batch, 512, 512, 6] => [batch, 256, 256, 64]
        l1 = tf.layers.conv2d(concat_inputs, filters=n_f, kernel_size=n_k, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=False)
        l1 = tf.maximum(alpha * l1, l1)

        # layer_2: [batch, 256, 256, 64] => [batch, 128, 128, 128]
        l2 = tf.layers.conv2d(l1, filters=n_f * 2, kernel_size=n_k, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=False)
        l2 = tf.layers.batch_normalization(inputs=l2, training=is_training)
        l2 = tf.maximum(alpha * l2, l2)

        # layer_3: [batch, 128, 128, 128] => [batch, 64, 64, 256]
        l3 = tf.layers.conv2d(l2, filters=n_f * 4, kernel_size=n_k, strides=2, padding='same',
                              kernel_initializer=w_init, use_bias=False)
        l3 = tf.layers.batch_normalization(inputs=l3, training=is_training)
        l3 = tf.maximum(alpha * l3, l3)

        # layer_4: [batch, 64, 64, 256] => [batch, 61, 61, 512]
        l4 = tf.layers.conv2d(l3, filters=n_f * 8, kernel_size=n_k, strides=1, padding='valid',
                              kernel_initializer=w_init, use_bias=False)

        # layer_5: [batch, 61, 61, 512] => [batch, 58, 58, 1]
        logits = tf.layers.conv2d(l4, filters=1, kernel_size=n_k, strides=1, padding='valid',
                                  kernel_initializer=w_init, use_bias=False)

        return logits