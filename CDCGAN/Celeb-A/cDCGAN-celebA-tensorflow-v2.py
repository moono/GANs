# prepare packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import time

# get data sets
from glob import glob
import helper

dataset_dir = '../../data_set/'

# get data
celebA_dataset = helper.Dataset(glob(os.path.join(dataset_dir, 'img_align_celeba/*.jpg')))
celebA_attr = helper.AttrParserCelebA(os.path.join(dataset_dir, 'celeba_anno/list_attr_celeba.txt'), attr_name='Male')


# our place holders
def model_inputs(image_width, image_height, image_channels, y_dim, z_dim):
    '''
    :param image_width: input image width. For MNIST 28. For celebA 64
    :param image_height: input image height. For MNIST 28. For celebA 64
    :param image_channel: input image channel. For MNIST 1. For celebA 3
    :param y_dim: label input size to discriminator & generator. For MNIST 10. For celebA 2 (Male or Female)
    :param z_dim: latent vector input size to generator. ex) 100
    '''
    inputs_x = tf.placeholder(tf.float32, [None, image_width, image_height, image_channels], name='inputs_x')
    inputs_y = tf.placeholder(tf.float32, [None, y_dim], name='inputs_y')
    inputs_y_reshaped = tf.placeholder(tf.float32, [None, image_width, image_height, y_dim], name='inputs_y_reshaped')
    inputs_z = tf.placeholder(tf.float32, [None, z_dim], name='inputs_z')

    return inputs_x, inputs_y, inputs_y_reshaped, inputs_z


# generator network structure
def generator(z, y, out_dim, reuse=False, initial_feature_size=1024, alpha=0.2, is_training=True):
    '''
    :param z: placeholder of latent vector
    :param y: placeholder of labels: For MNIST shape=[batch_size, 10], For celebA shape=[batch_size, 2]
    '''
    with tf.variable_scope('generator', reuse=reuse):
        # weight initializer
        w_init = tf.contrib.layers.xavier_initializer()

        # concatenate inputs
        # y_dim = y.get_shape().as_list()[1]
        concatenated_inputs = tf.concat(values=[z, y], axis=1)

        # 1. Fully connected layer (make 4x4x1024) & reshape to prepare first layer
        feature_map_size = initial_feature_size
        x1 = tf.layers.dense(inputs=concatenated_inputs,
                             units=4*4*feature_map_size,
                             activation=None,
                             use_bias=True,
                             kernel_initializer=w_init)
        x1 = tf.reshape(tensor=x1, shape=[-1, 4, 4, feature_map_size])
        x1 = tf.layers.batch_normalization(inputs=x1, training=is_training)
        x1 = tf.maximum(alpha * x1, x1)

        # 2. deconvolutional layer (make 8x8x512)
        feature_map_size = feature_map_size // 2
        x2 = tf.layers.conv2d_transpose(inputs=x1,
                                        filters=feature_map_size,
                                        kernel_size=5,
                                        strides=2,
                                        padding='same',
                                        activation=None,
                                        kernel_initializer=w_init)
        x2 = tf.layers.batch_normalization(inputs=x2, training=is_training)
        x2 = tf.maximum(alpha * x2, x2)

        # 3. deconvolutional layer (make 16x16x256)
        feature_map_size = feature_map_size // 2
        x3 = tf.layers.conv2d_transpose(inputs=x2,
                                        filters=feature_map_size,
                                        kernel_size=5,
                                        strides=2,
                                        padding='same',
                                        activation=None,
                                        kernel_initializer=w_init)
        x3 = tf.layers.batch_normalization(inputs=x3, training=is_training)
        x3 = tf.maximum(alpha * x3, x3)

        # 4. deconvolutional layer (make 32x32x128)
        feature_map_size = feature_map_size // 2
        x4 = tf.layers.conv2d_transpose(inputs=x3,
                                        filters=feature_map_size,
                                        kernel_size=5,
                                        strides=2,
                                        padding='same',
                                        activation=None,
                                        kernel_initializer=w_init)
        x4 = tf.layers.batch_normalization(inputs=x4, training=is_training)
        x4 = tf.maximum(alpha * x4, x4)

        # 4. Output layer, 64x64x3
        logits = tf.layers.conv2d_transpose(inputs=x4,
                                            filters=out_dim,
                                            kernel_size=5,
                                            strides=2,
                                            padding='same',
                                            activation=None,
                                            kernel_initializer=w_init)
        out = tf.tanh(logits)
    return out

# discriminator network structure
def discriminator(x, y_reshaped, reuse=False, initial_filter_size=128, alpha=0.2, is_training=True):
    '''
    :param x: placeholder of real or fake inputs. For MNIST shape=[batch_size, 28, 28, 1]. For celebA shape=[batch_size, 64, 64, 3]
    :param y_reshaped: placeholder of labels. For MNIST shape=[batch_size, 28, 28, 10]. For celebA shape=[batch_size, 64, 64, 2]
    '''
    with tf.variable_scope('discriminator', reuse=reuse):
        # weight initializer
        w_init = tf.contrib.layers.xavier_initializer()

        # x shape: 64x64x3
        # y_reshaped shape: 64x64x2
        # y_dim = y_reshaped.get_shape().as_list()[3]
        # concatenate inputs (makes 64x64x5)
        concatenated_inputs = tf.concat(axis=3, values=[x, y_reshaped])

        # 1. make 32x32x128
        filters = initial_filter_size
        x1 = tf.layers.conv2d(inputs=concatenated_inputs,
                              filters=filters,
                              kernel_size=5,
                              strides=2,
                              padding='same',
                              activation=None,
                              kernel_initializer=w_init)
        x1 = tf.maximum(alpha * x1, x1)

        # 2. make 16x16x256
        filters = filters * 2
        x2 = tf.layers.conv2d(inputs=x1,
                              filters=filters,
                              kernel_size=5,
                              strides=2,
                              padding='same',
                              activation=None,
                              kernel_initializer=w_init)
        x2 = tf.layers.batch_normalization(inputs=x2, training=True)
        x2 = tf.maximum(alpha * x2, x2)

        # 3. make 8x8x512
        filters = filters * 2
        x3 = tf.layers.conv2d(inputs=x2,
                              filters=filters,
                              kernel_size=5,
                              strides=2,
                              padding='same',
                              activation=None,
                              kernel_initializer=w_init)
        x3 = tf.layers.batch_normalization(inputs=x3, training=True)
        x3 = tf.maximum(alpha * x3, x3)

        # 4. make 4x4x1024
        filters = filters * 2
        x4 = tf.layers.conv2d(inputs=x3,
                              filters=filters,
                              kernel_size=5,
                              strides=2,
                              padding='same',
                              activation=None,
                              kernel_initializer=w_init)
        x4 = tf.layers.batch_normalization(inputs=x4, training=True)
        x4 = tf.maximum(alpha * x4, x4)

        # 5. flatten the layer
        flattend_layer = tf.reshape(tensor=x4, shape=[-1, 4*4*filters])
        logits = tf.layers.dense(inputs=flattend_layer,
                                 units=1,
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=w_init)
        out = tf.sigmoid(logits)
    return out, logits

def model_loss(inputs_x, inputs_y, inputs_y_reshaped, inputs_z, output_dim, alpha=0.2, smooth=0.0):
    # Generator network here (g_model is the generator output)
    g_model = generator(z=inputs_z, y=inputs_y, out_dim=output_dim, reuse=False, alpha=alpha, is_training=True)

    # Disriminator network here
    d_model_real, d_logits_real = discriminator(x=inputs_x, y_reshaped=inputs_y_reshaped, reuse=False, alpha=alpha, is_training=True)
    d_model_fake, d_logits_fake = discriminator(x=g_model, y_reshaped=inputs_y_reshaped, reuse=True, alpha=alpha, is_training=True)

    # Calculate losses
    real_labels = tf.ones_like(d_logits_real) * (1 - smooth) # label smoothing
    d_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=real_labels) )

    fake_labels = tf.zeros_like(d_logits_real)
    d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=fake_labels) )

    d_loss = d_loss_real + d_loss_fake

    gen_labels = tf.ones_like(d_logits_fake)
    g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=gen_labels) )

    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
    # Get the trainable_variables, split into G and D parts
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

'''
Build
'''
class cDCGAN:
    def __init__(self, y_size, z_size, learning_rate, alpha=0.2, beta1=0.5, smooth=0.0):
        # Size of input image to discriminator
        self.image_width = 64
        self.image_height = 64
        self.image_channels = 3

        self.y_size = y_size
        self.z_size = z_size

        tf.reset_default_graph()

        self.input_x, self.input_y, self.input_y_reshaped, self.input_z = model_inputs(self.image_width,
                                                                                       self.image_height,
                                                                                       self.image_channels,
                                                                                       y_size,
                                                                                       z_size)

        self.d_loss, self.g_loss = model_loss(self.input_x,
                                              self.input_y,
                                              self.input_y_reshaped,
                                              self.input_z,
                                              self.image_channels,
                                              alpha=alpha,
                                              smooth=smooth)

        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1)


# reshape y into appropriate input to D
def y_reshaper(y, width, height):
    new_y = np.zeros((y.shape[0], width, height, y.shape[1]))

    for i in range(y.shape[0]):
        new_y[i, :, :, :] = y[i,:] * np.ones((width, height, y.shape[1]))
    return new_y

# image save function
def save_generator_output(sess, fixed_y, fixed_z, input_y, input_z, out_dim, img_str, title):
    image_width = 64
    image_height = 64
    image_channels = 3
    n_rows = fixed_y.shape[0]
    n_cols = fixed_z.shape[0]

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols,n_rows), sharey=True, sharex=True)
    for ax_row, y_ in zip(axes, fixed_y):
        samples = sess.run( generator(input_z, input_y, out_dim, reuse=True, is_training=False),
                            feed_dict={input_y: y_, input_z: fixed_z})
        for ax, img in zip(ax_row, samples):
            ax.axis('off')
            ax.set_adjustable('box-forced')
            img = np.reshape(img, [image_width, image_height, image_channels])
            img = (((img - img.min()) * 255) / (img.max() - img.min())).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
    plt.savefig(img_str)
    plt.close(fig)


'''
Train
'''
def train(net, epochs, batch_size, print_every=50):
    y_size, z_size = net.y_size, net.z_size
    fixed_z = np.random.uniform(-1, 1, size=(10, z_size))
    fixed_y = np.zeros(shape=[y_size, 10, y_size])

    for c in range(y_size):
        fixed_y[c, :, c] = 1

    losses = []
    steps = 0

    save_every = 1 if epochs <= 10 else epochs // 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for ii in range(celebA_dataset.num_examples//batch_size):
            # for ii in range(1):
                steps += 1

                x_ = celebA_dataset.get_next_batch(batch_size)
                y_ = celebA_attr.get_next_batch(batch_size)

                # also reshape y for input_y_reshaped placeholder
                y_reshaped_ = y_reshaper(y_, net.image_width, net.image_height)

                # Sample random noise for G
                z_ = np.random.uniform(-1, 1, size=(batch_size, z_size))

                fd = {net.input_x: x_,
                      net.input_y: y_,
                      net.input_y_reshaped: y_reshaped_,
                      net.input_z: z_}
                # Run optimizers
                sess.run(net.d_opt, feed_dict=fd)
                sess.run(net.g_opt, feed_dict=fd)

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = net.d_loss.eval(fd)
                    train_loss_g = net.g_loss.eval(fd)

                    print("Epoch {}/{}...".format(e+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))

            if e % save_every == 0:
                # save generated images on every epochs
                image_fn = './assets/epoch_{:d}_tf.png'.format(e)
                image_title = 'epoch {:d}'.format(e)
                save_generator_output(sess, fixed_y, fixed_z, net.input_y, net.input_z, net.image_channels, image_fn, image_title)

    return losses

# Hyperparameters
y_size = 2 # labels: Female or Male
z_size = 100
learning_rate = 0.0002
batch_size = 32
epochs = 100
alpha = 0.2
beta1 = 0.5
smooth = 0.0

# Create the network
net = cDCGAN(y_size, z_size, learning_rate, alpha=alpha, beta1=beta1, smooth=smooth)

assets_dir = './assets/'
if not os.path.isdir(assets_dir):
    os.mkdir(assets_dir)

start_time = time.time()
losses = train(net, epochs, batch_size)
end_time = time.time()
total_time = end_time - start_time
print('Elapsed time: ', total_time)
# 20 epochs: 28756.74
# 100 epochs: 173213.42

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.savefig('./assets/losses_tf.png')

# create animated gif from result images
save_every = 1 if epochs <= 10 else epochs // 10
images = []
for e in range(epochs):
    if e % save_every == 0:
        image_fn = './assets/epoch_{:d}_tf.png'.format(e)
        images.append( imageio.imread(image_fn) )
imageio.mimsave('./assets/by_epochs_tf.gif', images, fps=3)
