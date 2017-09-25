'''
Below code is from
https://github.com/machrisaa/tensorflow-vgg
'''
import tensorflow as tf
import numpy as np

class Vgg16(object):
    def __init__(self, vgg16_npy_path=None, load_fc_too=False):
        if vgg16_npy_path is None:
            raise ValueError('Need vgg16.npy file to start')

        self.load_fc_too = load_fc_too
        self.im_size = 224
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.VGG_MEAN = [103.939, 116.779, 123.68]


    def run(self, rgb):
        # inputs: tf.placeholder of rgb image [batch, 224, 224, 3] values scaled [-1, 1]
        rgb_scaled = ((rgb + 1.0) / 2.0) * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])

        with tf.variable_scope('vgg16'):
            conv1_1 = self.conv_layer(bgr, "conv1_1")
            conv1_2 = self.conv_layer(conv1_1, "conv1_2")
            pool1 = self.max_pool(conv1_2, 'pool1')

            conv2_1 = self.conv_layer(pool1, "conv2_1")
            conv2_2 = self.conv_layer(conv2_1, "conv2_2")
            pool2 = self.max_pool(conv2_2, 'pool2')

            conv3_1 = self.conv_layer(pool2, "conv3_1")
            conv3_2 = self.conv_layer(conv3_1, "conv3_2")
            conv3_3 = self.conv_layer(conv3_2, "conv3_3")
            pool3 = self.max_pool(conv3_3, 'pool3')

            conv4_1 = self.conv_layer(pool3, "conv4_1")
            conv4_2 = self.conv_layer(conv4_1, "conv4_2")
            conv4_3 = self.conv_layer(conv4_2, "conv4_3")
            pool4 = self.max_pool(conv4_3, 'pool4')

            conv5_1 = self.conv_layer(pool4, "conv5_1")
            conv5_2 = self.conv_layer(conv5_1, "conv5_2")
            conv5_3 = self.conv_layer(conv5_2, "conv5_3")
            pool5 = self.max_pool(conv5_3, 'pool5')

            if self.load_fc_too:
                fc6 = self.fc_layer(pool5, "fc6")
                relu6 = tf.nn.relu(fc6)

                fc7 = self.fc_layer(relu6, "fc7")
                relu7 = tf.nn.relu(fc7)

                fc8 = self.fc_layer(relu7, "fc8")

                prob = tf.nn.softmax(fc8, name="prob")

        return conv1_2, conv2_2, conv3_3, conv4_3, conv5_3

    @staticmethod
    def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")