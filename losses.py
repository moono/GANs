import tensorflow as tf


# shorten cross entropy loss calculations
def celoss_ones(logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)))


def celoss_zeros(logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)))


def gan_loss_v1(d_real_logits, d_fake_logits):
    d_loss_real = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
    d_loss_fake = tf.reduce_mean(tf.nn.softplus(d_fake_logits))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(tf.nn.softplus(-d_fake_logits))
    return d_loss, g_loss


def gan_loss_v2(d_real_logits, d_fake_logits):
    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    d_loss = d_loss_real + d_loss_fake

    g_loss = celoss_ones(d_fake_logits)
    return d_loss, g_loss


def auxilary_classifier_loss(ac_real_logits, ac_fake_logits, labels):
    ac_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ac_real_logits, labels=labels))
    ac_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ac_fake_logits, labels=labels))
    ac_loss = ac_loss_real + ac_loss_fake
    return ac_loss


def wgan_loss(d_real_logits, d_fake_logits):
    d_loss_real = tf.reduce_mean(-d_real_logits)
    d_loss_fake = tf.reduce_mean(d_fake_logits)
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(-d_fake_logits)
    return d_loss, g_loss
