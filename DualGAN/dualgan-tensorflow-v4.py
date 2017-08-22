import tensorflow as tf
import numpy as np
import os
import time
import json

import helper
import ops

class DualGAN(object):
    def __init__(self, im_size, im_channel_u, im_channel_v):
        tf.reset_default_graph()

        #
        self.im_size, self.channel_u, self.channel_v = im_size, im_channel_u, im_channel_v

        self.n_filter_start = 64
        self.alpha = 0.2
        self.stddev = 0.02

        # loss related
        self.beta1 = 0.5
        self.lambda_u = 20.0
        self.lambda_v = 20.0
        self.learning_rate = 0.00005
        self.decay = 0.9

        # Build model
        self.input_u = tf.placeholder(tf.float32, [None, im_size, im_size, im_channel_u], name='input_u')
        self.input_v = tf.placeholder(tf.float32, [None, im_size, im_size, im_channel_v], name='input_v')

        # build graph
        self.gen_A_name = 'gen_A'
        self.gen_B_name = 'gen_B'
        self.dis_A_name = 'dis_A'
        self.dis_B_name = 'dis_B'

        # generator outputs
        w_init = tf.contrib.xavier_initializer()
        self.gen_A_out = ops.generator_unet_256(self.input_u, self.n_filter_start, self.channel_v,
                                                scope_name=self.gen_A_name, w_init_encoder=w_init,
                                                w_init_decoder=w_init, reuse=False)
        self.gen_B_out = ops.generator_unet_256(self.input_v, self.n_filter_start, self.channel_u,
                                                scope_name=self.gen_B_name, w_init_encoder=w_init,
                                                w_init_decoder=w_init, reuse=False)
        self.gen_AB_out = ops.generator_unet_256(self.gen_A_out, self.n_filter_start, self.channel_u,
                                                scope_name=self.gen_B_name, w_init_encoder=w_init,
                                                w_init_decoder=w_init, reuse=True)
        self.gen_BA_out = ops.generator_unet_256(self.gen_B_out, self.n_filter_start, self.channel_v,
                                                 scope_name=self.gen_A_name, w_init_encoder=w_init,
                                                 w_init_decoder=w_init, reuse=True)

        # discriminator outputs
        self.dis_A_real_logits = ops.discriminator_unet_256(self.input_v, self.n_filter_start,
                                                            scope_name=self.dis_A_name, w_init=w_init, reuse=False)
        self.dis_A_fake_logits = ops.discriminator_unet_256(self.gen_A_out, self.n_filter_start,
                                                            scope_name=self.dis_A_name, w_init=w_init, reuse=True)
        self.dis_B_real_logits = ops.discriminator_unet_256(self.input_u, self.n_filter_start,
                                                            scope_name=self.dis_B_name, w_init=w_init, reuse=False)
        self.dis_B_fake_logits = ops.discriminator_unet_256(self.gen_B_out, self.n_filter_start,
                                                            scope_name=self.dis_B_name, w_init=w_init, reuse=True)

        # discriminator losses
        self.dis_A_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_A_real_logits,
                                                    labels=tf.ones_like(self.dis_A_real_logits)))
        self.dis_A_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_A_fake_logits,
                                                    labels=tf.zeros_like(self.dis_A_fake_logits)))
        self.dis_B_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_B_real_logits,
                                                    labels=tf.ones_like(self.dis_B_real_logits)))
        self.dis_B_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_B_fake_logits,
                                                    labels=tf.zeros_like(self.dis_B_fake_logits)))
        self.d_loss = self.dis_A_real_loss + self.dis_A_fake_loss + self.dis_B_real_loss + self.dis_B_fake_loss

        # generator losses
        self.gen_A_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_A_fake_logits,
                                                    labels=tf.ones_like(self.dis_A_fake_logits)))
        self.gen_B_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_B_fake_logits,
                                                    labels=tf.ones_like(self.dis_B_fake_logits)))
        self.gen_A_l1_loss = tf.reduce_mean(tf.abs(self.gen_AB_out - self.input_u))
        self.gen_B_l1_loss = tf.reduce_mean(tf.abs(self.gen_BA_out - self.input_v))
        self.g_loss = self.gen_A_loss + self.gen_B_loss + \
                      (self.lambda_u * self.gen_A_l1_loss) + (self.lambda_v * self.gen_B_l1_loss)

        # optimizers
        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        u_d_vars = [var for var in t_vars if self.dis_A_name in var.name]
        v_d_vars = [var for var in t_vars if self.dis_B_name in var.name]
        u_g_vars = [var for var in t_vars if self.gen_A_name in var.name]
        v_g_vars = [var for var in t_vars if self.gen_B_name in var.name]
        self.d_vars = u_d_vars + v_d_vars
        self.g_vars = u_g_vars + v_g_vars

        print(len(self.d_vars))
        print(len(self.g_vars))

        # Optimizers
        self.d_train_opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay).\
            minimize(self.d_loss, var_list=self.d_vars)
        self.g_train_opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay). \
            minimize(self.g_loss, var_list=self.g_vars)

def train(net, dataset_name, train_data_loader, val_data_loader, epochs, batch_size, print_every=30, save_every=100):
    losses = []
    steps = 0

    # prepare saver for saving trained model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            # shuffle data randomly at every epoch
            train_data_loader.reset()
            # val_data_loader.reset()

            for ii in range(train_data_loader.n_images // batch_size):
                steps += 1

                batch_image_u, batch_image_v = train_data_loader.get_next_batch(batch_size)

                fd = {
                    net.input_u: batch_image_u,
                    net.input_v: batch_image_v
                }

                _ = sess.run(net.d_train_opt, feed_dict=fd)
                _ = sess.run(net.g_train_opt, feed_dict=fd)
                _ = sess.run(net.g_train_opt, feed_dict=fd)

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = net.d_loss.eval(fd)
                    train_loss_g = net.g_loss.eval(fd)

                    print("Epoch {}/{}...".format(e + 1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))

                if steps % save_every == 0:
                    # save generated images on every epochs
                    random_index = np.random.randint(0, val_data_loader.n_images)
                    test_image_u, test_image_v = val_data_loader.get_image_by_index(random_index)
                    gen_A_out, gen_AB_out = sess.run([net.gen_A_out, net.gen_AB_out], feed_dict={net.input_u: test_image_u})
                    gen_B_out, gen_BA_out = sess.run([net.gen_B_out, net.gen_BA_out], feed_dict={net.input_v: test_image_v})

                    image_fn = './assets/{:s}/epoch_{:d}-{:d}_tf.png'.format(dataset_name, e, steps)
                    helper.save_result(image_fn,
                                       test_image_u, gen_A_out, gen_AB_out,
                                       test_image_v, gen_B_out, gen_BA_out)

        ckpt_fn = './checkpoints/DualGAN-{:s}.ckpt'.format(dataset_name)
        saver.save(sess, ckpt_fn)

    return losses

def test(net, dataset_name, val_data_loader):
    ckpt_fn = './checkpoints/DualGAN-{:s}.ckpt'.format(dataset_name)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_fn)

        for ii in range(val_data_loader.n_images):
            test_image_u, test_image_v = val_data_loader.get_image_by_index(ii)

            gen_A_out, gen_AB_out = sess.run([net.gen_A_out, net.gen_AB_out], feed_dict={net.input_u: test_image_u})
            gen_B_out, gen_BA_out = sess.run([net.gen_B_out, net.gen_BA_out], feed_dict={net.input_v: test_image_v})

            image_fn = './assets/{:s}/{:s}_result_{:04d}_tf.png'.format(dataset_name, dataset_name, ii)
            helper.save_result(image_fn,
                               test_image_u, gen_A_out, gen_AB_out,
                               test_image_v, gen_B_out, gen_BA_out)

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
        net = DualGAN(im_size=im_size, im_channel_u=im_channel, im_channel_v=im_channel)

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

        else:
            # load train datasets
            val_data_loader = helper.Dataset(val_dataset_dir_u, val_dataset_dir_v, fn_ext,
                                             im_size, im_channel, im_channel, do_flip=False, do_shuffle=False)

            # validation
            test(net, dataset_name, val_data_loader)


if __name__ == '__main__':
    main()

