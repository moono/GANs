# prepare packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import imageio

# get data sets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../Data_sets/MNIST_data', one_hot=True)

# our place holders
def model_inputs(x_dim, y_dim, z_dim):
    '''
    :param x_dim: real input size to discriminator. For MNIST 784
    :param y_dim: label input size to discriminator & generator. For MNIST 10
    :param z_dim: latent vector input size to generator. ex) 100
    '''
    inputs_x = tf.placeholder(tf.float32, [None, x_dim], name='inputs_x')
    inputs_y = tf.placeholder(tf.float32, [None, y_dim], name='inputs_y')
    inputs_z = tf.placeholder(tf.float32, [None, z_dim], name='inputs_z')
    
    return inputs_x, inputs_y, inputs_z


# gennrator network structure
def generator(z, y, out_dim, n_units=128, reuse=False, alpha=0.2):
    '''
    :param z: placeholder of latent vector
    :param y: placeholder of labels
    '''
    with tf.variable_scope('generator', reuse=reuse):
        # weight initializer
        w_init = tf.contrib.layers.xavier_initializer()
        
        # concatenate inputs
        concatenated_inputs = tf.concat(axis=1, values=[z, y])
        
        h1 = tf.layers.dense(concatenated_inputs, n_units, activation=tf.nn.relu, use_bias=True, kernel_initializer=w_init)
        h1 = tf.maximum(alpha * h1, h1) # Leaky ReLU

        # Logits and tanh (-1~1) output
        logits = tf.layers.dense(h1, out_dim, activation=None, use_bias=True, kernel_initializer=w_init)
        out = tf.tanh(logits)
        
        return out

def discriminator(x, y, n_units=128, reuse=False, alpha=0.2):
    '''
    :param x: placeholder of real or fake inputs
    :param y: placeholder of labels
    '''
    with tf.variable_scope('discriminator', reuse=reuse):
        # weight initializer
        w_init = tf.contrib.layers.xavier_initializer()
        
        # concatenate inputs
        concatenated_inputs = tf.concat(axis=1, values=[x, y])
        
        h1 = tf.layers.dense(concatenated_inputs, n_units, activation=tf.nn.relu, use_bias=True, kernel_initializer=w_init)
        h1 = tf.maximum(alpha * h1, h1) # Leaky ReLU

        # Logits and sigmoid (0~1) output
        logits = tf.layers.dense(h1, 1, activation=None, use_bias=True, kernel_initializer=w_init)
        out = tf.sigmoid(logits)
        
        return out, logits

def model_loss(inputs_x, inputs_y, inputs_z, output_dim, alpha=0.2):
    # Generator network here (g_model is the generator output)
    g_model = generator(z=inputs_z, y=inputs_y, out_dim=output_dim, reuse=False, alpha=alpha)
    
    # Disriminator network here
    d_model_real, d_logits_real = discriminator(x=inputs_x, y=inputs_y, reuse=False, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(x=g_model, y=inputs_y, reuse=True, alpha=alpha)
    
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
    
    d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
    
    return d_train_opt, g_train_opt

# image save function
def save_generator_output(sess, fixed_z, fixed_y, input_y, input_z, out_dim, img_str, title):
    n_rows = fixed_y.shape[1] # number of labels
    n_cols = fixed_z.shape[0] # number of styles
    # print(fixed_z.shape)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5,5), sharey=True, sharex=True)
    for ax_row, y_ in zip(axes, fixed_y):
        samples = sess.run( generator(input_z, input_y, out_dim, reuse=True), 
                            feed_dict={input_z: fixed_z, input_y: y_})
        for ax, img in zip(ax_row, samples):
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
    def __init__(self, x_size, y_size, z_size, learning_rate, alpha=0.2, beta1=0.5):
        # wipe out previous graphs and make us to start building new graph from here
        tf.reset_default_graph()
        
        self.x_size = x_size

        self.input_x, self.input_y, self.input_z = model_inputs(x_size, y_size, z_size)
        
        self.d_loss, self.g_loss = model_loss(self.input_x, self.input_y, self.input_z, x_size, alpha=alpha)
        
        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1)


'''
Train
'''
def train(net, epochs, batch_size, x_size, y_size, z_size, print_every=50):
    fixed_z = np.random.uniform(-1, 1, size=(10, z_size))
    # create 0 ~ 9 labels
    fixed_y = np.zeros(shape=[y_size, 10, y_size])
    for c in range(y_size):
        fixed_y[c, :, c] = 1

    steps = 0
    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for ii in range(mnist.train.num_examples//batch_size):
                steps += 1
                
                # get batches
                x_, y_ = mnist.train.next_batch(batch_size)
                
                # Get images rescale to pass to D
                x_ = x_.reshape((batch_size, x_size))
                x_ = x_*2 -1
                
                # Sample random noise for G
                z_ = np.random.uniform(-1, 1, size=(batch_size, z_size))

                # Run optimizers
                sess.run(net.d_opt, feed_dict={net.input_x: x_, net.input_y: y_, net.input_z: z_})
                sess.run(net.g_opt, feed_dict={net.input_z: z_, net.input_y: y_})

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = net.d_loss.eval({net.input_x: x_, net.input_y: y_, net.input_z: z_})
                    train_loss_g = net.g_loss.eval({net.input_z: z_, net.input_y: y_})

                    print("Epoch {}/{}...".format(e+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))

            # save generated images on every epochs
            image_fn = './assets/epoch_{:d}_tf.png'.format(e)
            image_title = 'epoch {:d}'.format(e)
            save_generator_output(sess, fixed_z, fixed_y, net.input_y, net.input_z, net.x_size, image_fn, image_title)

    return losses

'''
hyper parameters
'''
x_size = 28 * 28 # 28x28 MNIST images flattened
y_size = 10 # lables: 0 ~ 9
z_size = 100
learning_rate = 0.001
epochs = 30
batch_size = 64
alpha = 0.2
smooth = 0.0
beta1 = 0.5

# Create the network
net = GAN(x_size, y_size, z_size, learning_rate, alpha=alpha, beta1=beta1)

assets_dir = './assets/'
if not os.path.isdir(assets_dir):
    os.mkdir(assets_dir)

start_time = time.time()
losses = train(net, epochs, batch_size, x_size, y_size, z_size)
end_time = time.time()
total_time = end_time - start_time
print('Elapsed time: ', total_time)
# 30 epochs: 285.55

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.savefig('./assets/losses_tf.png')
plt.close(fig)

# create animated gif from result images
images = []
for e in range(epochs):
    image_fn = './assets/epoch_{:d}_tf.png'.format(e)
    images.append( imageio.imread(image_fn) )
imageio.mimsave('./assets/by_epochs_tf.gif', images, fps=3)