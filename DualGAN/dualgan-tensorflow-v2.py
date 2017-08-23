import tensorflow as tf
import numpy as np
import os
import time
import json

import helper


def batch_norm(x,  name="batch_norm"):
    eps = 1e-6
    with tf.variable_scope(name):
        nchannels = x.get_shape()[3]
        scale = tf.get_variable("scale", [nchannels], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        center = tf.get_variable("center", [nchannels], initializer=tf.constant_initializer(0.0, dtype = tf.float32))
        ave, dev = tf.nn.moments(x, axes=[1,2], keep_dims=True)
        inv_dev = tf.rsqrt(dev + eps)
        normalized = (x-ave)*inv_dev * scale + center
        return normalized

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        desired_shape = conv.get_shape().as_list()
        desired_shape[0] = 1
        conv = tf.reshape(tf.nn.bias_add(conv, biases), desired_shape)

        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        desired_shape = deconv.get_shape().as_list()
        desired_shape[0] = 1
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), desired_shape)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def celoss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

class DualNet(object):
    def __init__(self, image_size=256, batch_size=1, fcn_filter_dim=64,
                 A_channels=3, B_channels=3, dataset_name='facades',
                 lambda_A=20., lambda_B=20., loss_metric='L1'):
        self.df_dim = fcn_filter_dim
        # self.flip = flip
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        # self.sess = sess
        # self.is_grayscale_A = (A_channels == 1)
        # self.is_grayscale_B = (B_channels == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.fcn_filter_dim = fcn_filter_dim
        self.A_channels = A_channels
        self.B_channels = B_channels
        self.loss_metric = loss_metric

        self.dataset_name = dataset_name
        # self.checkpoint_dir = checkpoint_dir

        # # directory name for output and logs saving
        # self.dir_name = "%s-img_sz_%s-fltr_dim_%d-%s-lambda_AB_%s_%s" % (
        #     self.dataset_name,
        #     self.image_size,
        #     self.fcn_filter_dim,
        #     self.loss_metric,
        #     self.lambda_A,
        #     self.lambda_B
        # )
        self.build_model()

    def build_model(self):
        ###    define place holders
        self.real_A = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size,
                                                  self.A_channels], name='real_A')
        self.real_B = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size,
                                                  self.B_channels], name='real_B')

        ###  define graphs
        self.A2B = self.A_g_net(self.real_A, reuse=False)
        self.B2A = self.B_g_net(self.real_B, reuse=False)
        self.A2B2A = self.B_g_net(self.A2B, reuse=True)
        self.B2A2B = self.A_g_net(self.B2A, reuse=True)

        if self.loss_metric == 'L1':
            self.A_loss = tf.reduce_mean(tf.abs(self.A2B2A - self.real_A))
            self.B_loss = tf.reduce_mean(tf.abs(self.B2A2B - self.real_B))
        elif self.loss_metric == 'L2':
            self.A_loss = tf.reduce_mean(tf.square(self.A2B2A - self.real_A))
            self.B_loss = tf.reduce_mean(tf.square(self.B2A2B - self.real_B))

        self.Ad_logits_fake = self.A_d_net(self.A2B, reuse=False)
        self.Ad_logits_real = self.A_d_net(self.real_B, reuse=True)
        self.Ad_loss_real = celoss(self.Ad_logits_real, tf.ones_like(self.Ad_logits_real))
        self.Ad_loss_fake = celoss(self.Ad_logits_fake, tf.zeros_like(self.Ad_logits_fake))
        self.Ad_loss = self.Ad_loss_fake + self.Ad_loss_real
        self.Ag_loss = celoss(self.Ad_logits_fake, labels=tf.ones_like(self.Ad_logits_fake)) + self.lambda_B * (self.B_loss)

        self.Bd_logits_fake = self.B_d_net(self.B2A, reuse=False)
        self.Bd_logits_real = self.B_d_net(self.real_A, reuse=True)
        self.Bd_loss_real = celoss(self.Bd_logits_real, tf.ones_like(self.Bd_logits_real))
        self.Bd_loss_fake = celoss(self.Bd_logits_fake, tf.zeros_like(self.Bd_logits_fake))
        self.Bd_loss = self.Bd_loss_fake + self.Bd_loss_real
        self.Bg_loss = celoss(self.Bd_logits_fake, tf.ones_like(self.Bd_logits_fake)) + self.lambda_A * (self.A_loss)

        self.d_loss = self.Ad_loss + self.Bd_loss
        self.g_loss = self.Ag_loss + self.Bg_loss
        ## define trainable variables
        t_vars = tf.trainable_variables()
        self.A_d_vars = [var for var in t_vars if 'A_d_' in var.name]
        self.B_d_vars = [var for var in t_vars if 'B_d_' in var.name]
        self.A_g_vars = [var for var in t_vars if 'A_g_' in var.name]
        self.B_g_vars = [var for var in t_vars if 'B_g_' in var.name]
        self.d_vars = self.A_d_vars + self.B_d_vars
        self.g_vars = self.A_g_vars + self.B_g_vars

        lr = 0.00005
        decay = 0.9
        self.d_optim = tf.train.RMSPropOptimizer(lr, decay=decay).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.RMSPropOptimizer(lr, decay=decay).minimize(self.g_loss, var_list=self.g_vars)

    def run_optim(self, sess, batch_A_imgs, batch_B_imgs, counter, start_time):
        _, Adfake, Adreal, Bdfake, Bdreal, Ad, Bd = sess.run(
            [self.d_optim, self.Ad_loss_fake, self.Ad_loss_real, self.Bd_loss_fake, self.Bd_loss_real, self.Ad_loss,
             self.Bd_loss],
            feed_dict={self.real_A: batch_A_imgs, self.real_B: batch_B_imgs})
        _, Ag, Bg, Aloss, Bloss = sess.run(
            [self.g_optim, self.Ag_loss, self.Bg_loss, self.A_loss, self.B_loss],
            feed_dict={self.real_A: batch_A_imgs, self.real_B: batch_B_imgs})

        _, Ag, Bg, Aloss, Bloss = sess.run(
            [self.g_optim, self.Ag_loss, self.Bg_loss, self.A_loss, self.B_loss],
            feed_dict={self.real_A: batch_A_imgs, self.real_B: batch_B_imgs})

        print("time: %4.4f, Ad: %.2f, Ag: %.2f, Bd: %.2f, Bg: %.2f,  U_diff: %.5f, V_diff: %.5f" \
              % (time.time() - start_time, Ad, Ag, Bd, Bg, Aloss, Bloss))
        print("Ad_fake: %.2f, Ad_real: %.2f, Bd_fake: %.2f, Bg_real: %.2f" % (Adfake, Adreal, Bdfake, Bdreal))

    def A_d_net(self, imgs, reuse=False):
        return self.discriminator(imgs, prefix='A_d_', reuse=reuse)

    def B_d_net(self, imgs, reuse=False):
        return self.discriminator(imgs, prefix='B_d_', reuse=reuse)

    def discriminator(self, image, prefix='A_d_', reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name=prefix + 'h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(batch_norm(conv2d(h0, self.df_dim * 2, name=prefix + 'h1_conv'), name=prefix + 'bn1'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name=prefix + 'h2_conv'), name=prefix + 'bn2'))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(
                batch_norm(conv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name=prefix + 'h3_conv'), name=prefix + 'bn3'))
            # h3 is (32 x 32 x self.df_dim*8)
            h4 = conv2d(h3, 1, d_h=1, d_w=1, name=prefix + 'h4')
            return h4

    def A_g_net(self, imgs, reuse=False):
        return self.fcn(imgs, prefix='A_g_', reuse=reuse)

    def B_g_net(self, imgs, reuse=False):
        return self.fcn(imgs, prefix='B_g_', reuse=reuse)

    def fcn(self, imgs, prefix=None, reuse=False):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            s = self.image_size
            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                s / 64), int(s / 128)

            # imgs is (256 x 256 x input_c_dim)
            e1 = conv2d(imgs, self.fcn_filter_dim, name=prefix + 'e1_conv')
            # e1 is (128 x 128 x self.fcn_filter_dim)
            e2 = batch_norm(conv2d(lrelu(e1), self.fcn_filter_dim * 2, name=prefix + 'e2_conv'), name=prefix + 'bn_e2')
            # e2 is (64 x 64 x self.fcn_filter_dim*2)
            e3 = batch_norm(conv2d(lrelu(e2), self.fcn_filter_dim * 4, name=prefix + 'e3_conv'), name=prefix + 'bn_e3')
            # e3 is (32 x 32 x self.fcn_filter_dim*4)
            e4 = batch_norm(conv2d(lrelu(e3), self.fcn_filter_dim * 8, name=prefix + 'e4_conv'), name=prefix + 'bn_e4')
            # e4 is (16 x 16 x self.fcn_filter_dim*8)
            e5 = batch_norm(conv2d(lrelu(e4), self.fcn_filter_dim * 8, name=prefix + 'e5_conv'), name=prefix + 'bn_e5')
            # e5 is (8 x 8 x self.fcn_filter_dim*8)
            e6 = batch_norm(conv2d(lrelu(e5), self.fcn_filter_dim * 8, name=prefix + 'e6_conv'), name=prefix + 'bn_e6')
            # e6 is (4 x 4 x self.fcn_filter_dim*8)
            e7 = batch_norm(conv2d(lrelu(e6), self.fcn_filter_dim * 8, name=prefix + 'e7_conv'), name=prefix + 'bn_e7')
            # e7 is (2 x 2 x self.fcn_filter_dim*8)
            e8 = batch_norm(conv2d(lrelu(e7), self.fcn_filter_dim * 8, name=prefix + 'e8_conv'), name=prefix + 'bn_e8')
            # e8 is (1 x 1 x self.fcn_filter_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                                                     [self.batch_size, s128, s128, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd1', with_w=True)
            d1 = tf.nn.dropout(batch_norm(self.d1, name=prefix + 'bn_d1'), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.fcn_filter_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                                                     [self.batch_size, s64, s64, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd2', with_w=True)
            d2 = tf.nn.dropout(batch_norm(self.d2, name=prefix + 'bn_d2'), 0.5)

            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.fcn_filter_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                                                     [self.batch_size, s32, s32, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd3', with_w=True)
            d3 = tf.nn.dropout(batch_norm(self.d3, name=prefix + 'bn_d3'), 0.5)

            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.fcn_filter_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                                                     [self.batch_size, s16, s16, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd4', with_w=True)
            d4 = batch_norm(self.d4, name=prefix + 'bn_d4')

            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.fcn_filter_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                                                     [self.batch_size, s8, s8, self.fcn_filter_dim * 4],
                                                     name=prefix + 'd5', with_w=True)
            d5 = batch_norm(self.d5, name=prefix + 'bn_d5')
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.fcn_filter_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                                                     [self.batch_size, s4, s4, self.fcn_filter_dim * 2],
                                                     name=prefix + 'd6', with_w=True)
            d6 = batch_norm(self.d6, name=prefix + 'bn_d6')
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.fcn_filter_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                                                     [self.batch_size, s2, s2, self.fcn_filter_dim], name=prefix + 'd7',
                                                     with_w=True)
            d7 = batch_norm(self.d7, name=prefix + 'bn_d7')
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.fcn_filter_dim*1*2)

            if prefix == 'B_g_':
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, self.A_channels],
                                                         name=prefix + 'd8', with_w=True)
            elif prefix == 'A_g_':
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, self.B_channels],
                                                         name=prefix + 'd8', with_w=True)
                # d8 is (256 x 256 x output_c_dim)
            return tf.nn.tanh(self.d8)

def train(net, dataset_name, train_data_loader, val_data_loader, epochs, batch_size, print_every=30, save_every=100):
    losses = []
    steps = 0

    # prepare saver for saving trained model
    saver = tf.train.Saver()

    start_time = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            # shuffle data randomly at every epoch
            train_data_loader.reset()
            # val_data_loader.reset()

            for ii in range(train_data_loader.n_images // batch_size):
                steps += 1

                batch_image_u, batch_image_v = train_data_loader.get_next_batch(batch_size)

                net.run_optim(sess, batch_image_u, batch_image_v, steps, start_time)

                if steps % save_every == 0:
                    # save generated images on every epochs
                    random_index = np.random.randint(0, val_data_loader.n_images)
                    test_image_u, test_image_v = val_data_loader.get_image_by_index(random_index)
                    fd_val = {
                        net.real_A: test_image_u,
                        net.real_B: test_image_v
                    }
                    g_image_u_to_v_to_u, g_image_u_to_v = sess.run([net.A2B2A, net.A2B], feed_dict=fd_val)
                    g_image_v_to_u_to_v, g_image_v_to_u = sess.run([net.B2A2B, net.B2A], feed_dict=fd_val)

                    image_fn = './assets/{:s}/epoch_{:d}-{:d}_tf.png'.format(dataset_name, e, steps)
                    helper.save_result(image_fn,
                                       test_image_u, g_image_u_to_v, g_image_u_to_v_to_u,
                                       test_image_v, g_image_v_to_u, g_image_v_to_u_to_v)

        ckpt_fn = './checkpoints/DualGAN-{:s}.ckpt'.format(dataset_name)
        saver.save(sess, ckpt_fn)

    return losses

# def test(net, dataset_name, val_data_loader):
#     ckpt_fn = './checkpoints/DualGAN-{:s}.ckpt'.format(dataset_name)
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         saver.restore(sess, ckpt_fn)
#
#         # run on u set
#         for ii in range(val_data_loader.n_images):
#             test_image_u = val_data_loader.get_image_by_index_u(ii)
#
#             g_image_u_to_v, g_image_u_to_v_to_u = sess.run([net.g_model_u2v, net.g_model_u2v2u],
#                                                            feed_dict={net.input_u: test_image_u})
#
#             image_fn = './assets/{:s}/{:s}_result_u_{:04d}_tf.png'.format(dataset_name, dataset_name, ii)
#             helper.save_result_single_row(image_fn, test_image_u, g_image_u_to_v, g_image_u_to_v_to_u)
#
#         # run on v set
#         for ii in range(val_data_loader.n_images):
#             test_image_v = val_data_loader.get_image_by_index_v(ii)
#             g_image_v_to_u, g_image_v_to_u_to_v = sess.run([net.g_model_v2u, net.g_model_v2u2v],
#                                                            feed_dict={net.input_v: test_image_v})
#
#             image_fn = './assets/{:s}/{:s}_result_v_{:04d}_tf.png'.format(dataset_name, dataset_name, ii)
#             helper.save_result_single_row(image_fn, test_image_v, g_image_v_to_u, g_image_v_to_u_to_v)

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
        net = DualNet(image_size=im_size, A_channels=im_channel, B_channels=im_channel, dataset_name=dataset_name)

        if not is_test:
            # load train & validation datasets
            train_data_loader = helper.Dataset(train_dataset_dir_u, train_dataset_dir_v, fn_ext,
                                               im_size, im_channel, im_channel, do_flip=do_flip, do_shuffle=True)
            val_data_loader = helper.Dataset(val_dataset_dir_u, val_dataset_dir_v, fn_ext,
                                             im_size, im_channel, im_channel, do_flip=False, do_shuffle=False)

            # start training
            start_time = time.time()
            losses = train(net, dataset_name, train_data_loader, val_data_loader, epochs, batch_size)
            end_time = time.time()
            total_time = end_time - start_time
            test_result_str = '[Training]: Data: {:s}, Epochs: {:3f}, Batch_size: {:2d}, Elapsed time: {:3f}\n'.format(
                dataset_name, epochs, batch_size, total_time)
            print(test_result_str)

            with open('./assets/test_summary.txt', 'a') as f:
                f.write(test_result_str)

        # else:
        #     # load train datasets
        #     val_data_loader = helper.Dataset(val_dataset_dir_u, val_dataset_dir_v, fn_ext,
        #                                      im_size, im_channel, im_channel, do_flip=False, do_shuffle=False)
        #
        #     # validation
        #     test(net, dataset_name, val_data_loader)

# def test1():
#     fn_ext = 'jpg'
#     dataset_name = 'sketch-photo'
#     val_dir_u = '../data_set/DualGAN/sketch-photo/val/A'
#     val_dir_v = '../data_set/DualGAN/sketch-photo/val/B'
#     im_size = 256
#     im_channel = 1
#
#     val_data_loader = helper.Dataset(val_dir_u, val_dir_v, fn_ext,
#                                      im_size, im_channel, im_channel, do_flip=False, do_shuffle=False)
#     net = DualGAN(im_size=im_size, im_channel_u=im_channel, im_channel_v=im_channel)
#     test(net, dataset_name, val_data_loader)

if __name__ == '__main__':
    main()
