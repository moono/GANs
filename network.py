import tensorflow as tf


def generator(z, y=None, embed_y=False, is_training=True, use_bn=True):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        n_filter = 64
        n_kernel = 5

        # 0. concatenate inputs
        # z: [batch size, 100], y: [batch size, 10]
        if y is not None:
            if embed_y:
                z_dim = z.get_shape().as_list()[-1]
                y_dim = y.get_shape().as_list()[-1]
                with tf.variable_scope('embed_y'):
                    w = tf.get_variable('weight', shape=[y_dim, z_dim], dtype=tf.float32,
                                        initializer=tf.initializers.random_normal())
                    y = tf.matmul(y, w)
                    inputs = tf.concat([z, y], axis=1)
            else:
                inputs = tf.concat([z, y], axis=1)
        else:
            inputs = z

        # 1. reshape z-vector to fit as 2d shape image with fully connected layer
        l1 = tf.layers.dense(inputs, units=4 * 4 * n_filter * 4)
        if use_bn:
            l1 = tf.layers.batch_normalization(l1, training=is_training)
        l1 = tf.nn.leaky_relu(l1)
        l1 = tf.reshape(l1, shape=[-1, 4, 4, n_filter * 4])

        # 2. layer2 - [batch size, 4, 4, 256] ==> [batch size, 7, 7, 128]
        l2 = tf.layers.conv2d_transpose(l1, filters=n_filter * 2, kernel_size=n_kernel, strides=2, padding='same')
        if use_bn:
            l2 = tf.layers.batch_normalization(l2, training=is_training)
        l2 = tf.nn.leaky_relu(l2)
        l2 = l2[:, :7, :7, :]   # match shape...

        # 3. layer3 - [batch size, 7, 7, 128] ==> [batch size, 14, 14, 64]
        l3 = tf.layers.conv2d_transpose(l2, filters=n_filter, kernel_size=n_kernel, strides=2, padding='same')
        if use_bn:
            l3 = tf.layers.batch_normalization(l3, training=is_training)
        l3 = tf.nn.leaky_relu(l3)

        # 4. layer4 - [batch size, 14, 14, 64] ==> [batch size, 28, 28, 1]
        l4 = tf.layers.conv2d_transpose(l3, filters=1, kernel_size=n_kernel, strides=2, padding='same')
        out = tf.tanh(l4)
        return out


def discriminator(x, y=None, y_conditioning=False, is_training=True, use_bn=True):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        n_filter = 64
        n_kernel = 5

        # 0. concatenate inputs
        # x: [batch size, 28, 28, 1], y: [batch size, 10]
        # make y as same dimension as x first
        if y is not None and y_conditioning is False:
            y_tiled = tf.expand_dims(y, axis=1)
            y_tiled = tf.expand_dims(y_tiled, axis=1)
            y_tiled = tf.tile(y_tiled, multiples=[1, 28, 28, 1])
            inputs = tf.concat([x, y_tiled], axis=3)
        else:
            inputs = x

        # 1. layer 1 - [batch size, 28, 28, 1] ==> [batch size, 14, 14, 64]
        l1 = tf.layers.conv2d(inputs, filters=n_filter, kernel_size=n_kernel, strides=2, padding='same')
        l1 = tf.nn.leaky_relu(l1)

        # 2. layer 2 - [batch size, 14, 14, 64] ==> [batch size, 7, 7, 128]
        l2 = tf.layers.conv2d(l1, filters=n_filter * 2, kernel_size=n_kernel, strides=2, padding='same')
        if use_bn:
            l2 = tf.layers.batch_normalization(l2, training=is_training)
        l2 = tf.nn.leaky_relu(l2)

        # 3. layer 3 - [batch size, 7, 7, 128] ==> [batch size, 4, 4, 256]
        l3 = tf.layers.conv2d(l2, filters=n_filter * 4, kernel_size=n_kernel, strides=2, padding='same')
        if use_bn:
            l3 = tf.layers.batch_normalization(l3, training=is_training)
        l3 = tf.nn.leaky_relu(l3)

        # 4. flatten layer & fully connected layer
        l4 = tf.layers.flatten(l3)

        # final logits
        logits = tf.layers.dense(l4, units=1)

        if y is not None and y_conditioning is True:
            with tf.variable_scope('label_conditioning'):
                logits = logits * y
                logits = tf.reduce_sum(logits, axis=1, keepdims=True)

        return logits, l4


def classifier(x, out_dim, is_training=True, use_bn=True):
    with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
        # 1. layer 1 - fully connected layer
        l1 = tf.layers.dense(x, 128)
        if use_bn:
            l1 = tf.layers.batch_normalization(l1, training=is_training)
        l1 = tf.nn.leaky_relu(l1)

        # final logits
        logits = tf.layers.dense(l1, out_dim)

        return logits
