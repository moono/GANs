import tensorflow as tf


def embed_label(label, label_dim, channel_dim):
    with tf.variable_scope('embed_label'):
        w = tf.get_variable('weight', shape=[label_dim, channel_dim], dtype=tf.float32,
                            initializer=tf.initializers.random_normal())
        y = tf.matmul(label, w)
    return y


def generator(z, y=None, embed_y=False, is_training=True, use_bn=True):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        n_filter = 64
        n_kernel = 5

        # 0. concatenate inputs
        # z: [batch size, 100], y: [batch size, 10]
        if y is not None:
            if embed_y:
                embedded_y = embed_label(y, y.get_shape().as_list()[-1], z.get_shape().as_list()[-1])
                inputs = tf.concat([z, embedded_y], axis=1)
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


def discriminator(x, y=None, embed_y=False, is_training=True, use_bn=True):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        n_filter = 64
        n_kernel = 5

        # 0. concatenate inputs
        # x: [batch size, 28, 28, 1], y: [batch size, 10]
        # make y as same dimension as x first
        if y is not None and embed_y is False:
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

        if y is not None and embed_y is True:
            with tf.variable_scope('projection_discriminator'):
                # global average pooling
                h = tf.reduce_mean(l3, axis=[1, 2])

                # compute logit
                logits = tf.layers.dense(h, units=1)

                # embed label to last cnn feature size
                embedded_y = embed_label(y, y.get_shape().as_list()[-1], h.get_shape().as_list()[-1])

                # apply inner product between embedded label & last cnn feature
                logits = logits + tf.reduce_sum(embedded_y * h, axis=1, keepdims=True)
                l4 = h

            # with tf.variable_scope('label_conditioning'):
            #     l4 = tf.layers.flatten(l3)
            #     logits = tf.layers.dense(l4, units=y.get_shape().as_list()[-1])
            #
            #     conditioned = logits * y
            #     logits = tf.reduce_sum(conditioned, axis=1, keepdims=True)
        else:
            # 4. flatten layer & fully connected layer
            l4 = tf.layers.flatten(l3)

            # final logits
            logits = tf.layers.dense(l4, units=1)

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


def projection_discriminator(x, y, is_training=True, use_bn=True):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        n_filter = 64
        n_kernel = 5

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

        # global average pooling
        h = tf.reduce_mean(l3, axis=[1, 2])

        # compute logit
        logits = tf.layers.dense(h, units=1)

        # embed label to last cnn feature size
        embedded_y = embed_label(y, y.get_shape().as_list()[-1], h.get_shape().as_list()[-1])

        # apply inner product between embedded label & last cnn feature
        logits = logits + tf.reduce_sum(embedded_y * h, axis=1, keepdims=True)

        return logits


def label_conditioning_discriminator(x, y, is_training=True, use_bn=True):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        n_filter = 64
        n_kernel = 5

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
        logits = tf.layers.dense(l4, units=y.get_shape().as_list()[-1])

        # condition logits with input label
        logits = tf.reduce_sum(logits * y, axis=1, keepdims=True)
        return logits


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        norm = tf.reduce_mean(tf.square(x), axis=1, keepdims=True)
        x = x * tf.rsqrt(norm + epsilon)
    return x


def discriminator_block(x, y, n_f, n_k, is_training, use_bn, scope_index):
    with tf.variable_scope('block_{}'.format(scope_index)):
        x = tf.layers.conv2d(x, filters=n_f, kernel_size=n_k, strides=2, padding='same')
        if use_bn:
            x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        h = tf.reduce_mean(x, axis=[1, 2])
        embedded_y = embed_label(y, y.get_shape().as_list()[-1], h.get_shape().as_list()[-1])
        embedded_y = pixel_norm(embedded_y)
        projected = tf.reduce_sum(embedded_y * h, axis=1, keepdims=True)
    return x, projected


def new_discriminator(x, y, is_training=True, use_bn=True):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        n_filter = 64
        n_kernel = 5

        # 1. layer 1 - [batch size, 28, 28, 1] ==> [batch size, 14, 14, 64]
        l1, p1 = discriminator_block(x, y, n_filter * 2, n_kernel, is_training, False, scope_index=1)

        # 2. layer 2 - [batch size, 14, 14, 64] ==> [batch size, 7, 7, 128]
        l2, p2 = discriminator_block(l1, y, n_filter * 2, n_kernel, is_training, use_bn, scope_index=2)

        # 3. layer 3 - [batch size, 7, 7, 128] ==> [batch size, 4, 4, 256]
        l3, p3 = discriminator_block(l2, y, n_filter * 4, n_kernel, is_training, use_bn, scope_index=3)

        # 4. layer 4 - [batch size, 4, 4, 256] ==> [batch size, 2, 2, 256]
        l4, p4 = discriminator_block(l3, y, n_filter * 4, n_kernel, is_training, use_bn, scope_index=4)

        # 5. layer 5 - [batch size, 2, 2, 256] ==> [batch size, 1, 1, 256]
        l5, p5 = discriminator_block(l4, y, n_filter * 4, n_kernel, is_training, use_bn, scope_index=5)

        # compute logit
        logits = tf.layers.dense(tf.reshape(l5, shape=[-1, n_filter * 4]), units=1)

        # apply inner product between embedded label & last cnn feature
        hyper_feature = tf.concat([p1, p2, p3, p4, p5], axis=1)
        logits = logits + tf.reduce_sum(hyper_feature, axis=1, keepdims=True)
        return logits


def main():
    x = tf.constant(0, dtype=tf.float32, shape=[32, 28, 28, 1])
    y = tf.constant(0, dtype=tf.float32, shape=[32, 10])

    # logits = new_discriminator(x, y, is_training=True, use_bn=True)
    logits = projection_discriminator(x, y, is_training=True, use_bn=True)
    return


if __name__ == '__main__':
    main()
