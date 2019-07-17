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


# https://github.com/taki0112/Tensorflow-Cookbook/blob/master/ops.py#L610
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable('u', [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm



def projectiond_embedding(label, label_dim, channel_dim):
    with tf.variable_scope('projectiond_embedding_label'):
        w = tf.get_variable('weight', shape=[label_dim, channel_dim], dtype=tf.float32)
        w = spectral_norm(w)
        y = tf.matmul(label, w)
    return y


# https://github.com/pfnet-research/sngan_projection
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

        # l3 = [batch size, 4, 4, 256]
        # phi = [batch size, 256]
        phi = tf.reduce_sum(l3, axis=[1, 2])

        # psi = [batch size, 1]
        psi = tf.layers.dense(phi, units=1)

        # embed label to last cnn feature size: [batch size, 256]
        embedded_y = projectiond_embedding(y, y.get_shape().as_list()[-1], phi.get_shape().as_list()[-1])

        # logits: [batch size, 1]
        logits = psi + tf.reduce_sum(embedded_y * phi, axis=1, keepdims=True)
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


def main():
    x = tf.constant(0, dtype=tf.float32, shape=[32, 28, 28, 1])
    y = tf.constant(0, dtype=tf.float32, shape=[32, 10])
    logits = projection_discriminator(x, y, is_training=True, use_bn=True)
    return


if __name__ == '__main__':
    main()
