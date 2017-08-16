# prepare packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import time


# get data sets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../data_set/MNIST_data')

# input placeholders
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    
    return inputs_real, inputs_z

def generator(z, output_dim, reuse=False, initial_feature_size=512, alpha=0.2, is_training=True):
    with tf.variable_scope('generator', reuse=reuse):
        feature_map_size = initial_feature_size
        
        # 1. Fully connected layer (make 3x3x512)
        first_layer_units = 3 * 3 * feature_map_size
        x1 = tf.layers.dense(inputs=z, units=first_layer_units, activation=None, use_bias=True)
        # reshape as convolutional layer
        x1 = tf.reshape(tensor=x1, shape=[-1, 3, 3, feature_map_size])
        # add batch normalization
        x1 = tf.layers.batch_normalization(inputs=x1, training=is_training)
        # add reaky relu activation
        x1 = tf.maximum(alpha * x1, x1)
        
        # 2. convolutional layer (make 7x7x256)
        feature_map_size = feature_map_size // 2
        x2 = tf.layers.conv2d_transpose(inputs=x1, filters=feature_map_size, kernel_size=3, strides=2, padding='valid')
        x2 = tf.layers.batch_normalization(inputs=x2, training=is_training)
        x2 = tf.maximum(alpha * x2, x2)
        
        # 3. convolutional layer (make 14x14x128)
        feature_map_size = feature_map_size // 2
        x3 = tf.layers.conv2d_transpose(inputs=x2, filters=feature_map_size, kernel_size=5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(inputs=x3, training=is_training)
        x3 = tf.maximum(alpha * x3, x3)
        
        # Output layer, 28x28x1
        logits = tf.layers.conv2d_transpose(inputs=x3, filters=output_dim, kernel_size=5, strides=2, padding='same')
        out = tf.tanh(logits)
        
        return out

def discriminator(x, reuse=False, initial_filter_size=64, alpha=0.2, is_training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        # starting variable
        filters = initial_filter_size
        # Input layer is 28x28x1
        
        # make 14x14x64
        x1 = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=5, strides=2, padding='same')
        x1 = tf.maximum(alpha * x1, x1)
        
        # make 7x7x128
        filters = filters * 2
        x2 = tf.layers.conv2d(inputs=x1, filters=filters, kernel_size=5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(inputs=x2, training=is_training)
        x2 = tf.maximum(alpha * x2, x2)
        
        # make 4x4x256
        filters = filters * 2
        x3 = tf.layers.conv2d(inputs=x2, filters=filters, kernel_size=5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(inputs=x3, training=is_training)
        x3 = tf.maximum(alpha * x3, x3)
        
        # flatten the layer
        flattened_layer = tf.reshape(tensor=x3, shape=[-1, 4*4*filters])
        logits = tf.layers.dense(inputs=flattened_layer, units=1)
        out = tf.sigmoid(logits)
        
        return out, logits

def model_loss(input_real, input_z, output_dim, alpha=0.2):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    g_model = generator(input_z, output_dim, alpha=alpha)
    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, alpha=alpha)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


'''
Build
'''
class DCGAN:
    def __init__(self, real_size, z_size, learning_rate, alpha=0.2, beta1=0.5):
        tf.reset_default_graph()
        
        self.input_real, self.input_z = model_inputs(real_size, z_size)
        
        self.d_loss, self.g_loss = model_loss(self.input_real, self.input_z, real_size[2], alpha=0.2)
        
        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1)

# image save function
def save_generator_output(sess, fixed_z, input_z, out_channel_dim, img_str, title):
    n_images = fixed_z.shape[0]
    n_rows = np.sqrt(n_images).astype(np.int32)
    n_cols = np.sqrt(n_images).astype(np.int32)
    
    samples = sess.run(
        generator(input_z, out_channel_dim, reuse=True, is_training=False),
        feed_dict={input_z: fixed_z})
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5,5), sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        ax.imshow(img.reshape((28,28)), cmap='Greys_r', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
    plt.savefig(img_str)
    plt.close(fig)


'''
Train
'''
def train(net, epochs, batch_size, print_every=10):
    fixed_z = np.random.uniform(-1, 1, size=(25, z_size))

    losses = []
    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for ii in range(mnist.train.num_examples//batch_size):
                steps += 1
                
                # no need labels
                batch_x, _ = mnist.train.next_batch(batch_size)
                
                # Get images rescale to pass to D
                batch_images = batch_x.reshape(-1, 28, 28, 1)
                batch_images = batch_images*2 -1
                
                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

                # Run optimizers
                sess.run(net.d_opt, feed_dict={net.input_real: batch_images, net.input_z: batch_z})
                sess.run(net.g_opt, feed_dict={net.input_z: batch_z, net.input_real: batch_images})

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = net.d_loss.eval({net.input_z: batch_z, net.input_real: batch_images})
                    train_loss_g = net.g_loss.eval({net.input_z: batch_z})

                    print("Epoch {}/{}...".format(e+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))

            # save generated images on every epochs
            image_fn = './assets/epoch_{:d}_tf.png'.format(e)
            image_title = 'epoch {:d}'.format(e)
            save_generator_output(sess, fixed_z, net.input_z, 1, image_fn, image_title)

    return losses


# Hyperparameters
real_size = (28,28,1)
z_size = 100
learning_rate = 0.0002
batch_size = 128
epochs = 30
alpha = 0.2
beta1 = 0.5

# Create the network
net = DCGAN(real_size, z_size, learning_rate, alpha=alpha, beta1=beta1)

assets_dir = './assets/'
if not os.path.isdir(assets_dir):
    os.mkdir(assets_dir)

start_time = time.time()
losses = train(net, epochs, batch_size)
end_time = time.time()
total_time = end_time - start_time
print('Elapsed time: ', total_time)
# 30 epochs: 718.14

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.savefig('./assets/losses_tf.png')

# create animated gif from result images
images = []
for e in range(epochs):
    image_fn = './assets/epoch_{:d}_tf.png'.format(e)
    images.append( imageio.imread(image_fn) )
imageio.mimsave('./assets/by_epochs_tf.gif', images, fps=3)

