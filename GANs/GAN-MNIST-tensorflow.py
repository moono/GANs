import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import imageio

# get data sets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../Data_sets/MNIST_data')

# our place holders
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, [None, real_dim], name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, [None, z_dim], name='inputs_z')
    
    return inputs_real, inputs_z

# gennrator network structure
def generator(z, out_dim, n_units=128, reuse=False,  alpha=0.2):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        h1 = tf.layers.dense(z, n_units, activation=None, kernel_initializer=w_init)
        h1 = tf.maximum(alpha * h1, h1) # Leaky ReLU
        
        # Logits and tanh(-1~1) output
        logits = tf.layers.dense(h1, out_dim, activation=None, kernel_initializer=w_init)
        out = tf.tanh(logits)
        
        return out

def discriminator(x, n_units=128, reuse=False, alpha=0.2):
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        h1 = tf.layers.dense(x, n_units, activation=None, kernel_initializer=w_init)
        h1 = tf.maximum(alpha * h1, h1) # Leaky ReLU

        # Logits and sigmoid(0~1) output
        logits = tf.layers.dense(h1, 1, activation=None, kernel_initializer=w_init)
        out = tf.sigmoid(logits)
        
        return out, logits

def model_loss(input_real, input_z, output_dim, alpha=0.2):
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
    d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

# image save function
def save_generator_output(sess, fixed_z, input_z, out_dim, img_str, title):
    n_images = fixed_z.shape[0]
    n_rows = np.sqrt(n_images).astype(np.int32)
    n_cols = np.sqrt(n_images).astype(np.int32)
    
    samples = sess.run(
        generator(input_z, out_dim, reuse=True),
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
Build
'''
class GAN:
    def __init__(self, input_size, z_size, learning_rate, alpha=0.2, beta1=0.5):
        # wipe out previous graphs and make us to start building new graph from here
        tf.reset_default_graph()
        
        self.input_size, self.z_size = input_size, z_size

        self.input_real, self.input_z = model_inputs(input_size, z_size)
        
        self.d_loss, self.g_loss = model_loss(self.input_real, self.input_z, input_size, alpha=0.2)
        
        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1)


'''
Train
'''
def train(net, epochs, batch_size, print_every=50):
    fixed_z = np.random.uniform(-1, 1, size=(25, net.z_size))

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
                batch_images = batch_x.reshape((batch_size, net.input_size))
                batch_images = batch_images*2 -1
                
                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, net.z_size))

                # Run optimizers
                sess.run(net.d_opt, feed_dict={net.input_real: batch_images, net.input_z: batch_z})
                sess.run(net.g_opt, feed_dict={net.input_z: batch_z})

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
            save_generator_output(sess, fixed_z, net.input_z, net.input_size, image_fn, image_title)

    return losses

def main():
    '''
    hyper parameters
    '''
    input_size = 28 * 28  # 28x28 MNIST images flattened
    z_size = 100
    learning_rate = 0.002
    epochs = 30
    batch_size = 128
    alpha = 0.2
    smooth = 0.0
    beta1 = 0.5

    # Create the network
    net = GAN(input_size, z_size, learning_rate, alpha=alpha, beta1=beta1)

    assets_dir = './assets/'
    if not os.path.isdir(assets_dir):
        os.mkdir(assets_dir)

    start_time = time.time()
    losses = train(net, epochs, batch_size)
    end_time = time.time()
    total_time = end_time - start_time
    print('Elapsed time: ', total_time)
    # 30 epochs: 94.36

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
        images.append(imageio.imread(image_fn))
    imageio.mimsave('./assets/by_epochs_tf.gif', images, fps=3)

if __name__ == "__main__":
    main()