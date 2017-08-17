import torch
import torch.utils.data
import torch.nn.functional as torch_func
from torch.nn import init
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import imageio


# initialize weight with xavier
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0)

# define Generator
class Generator(torch.nn.Module):
    def __init__(self, z_size, y_size, n_hidden=128, n_out=728, alpha=0.2):
        super().__init__()
        self.alpha = alpha

        # define 2 fully connected layers
        input_size = z_size + y_size
        self.fc1 = torch.nn.Linear(input_size, n_hidden, bias=True)
        self.fc2 = torch.nn.Linear(n_hidden, n_out, bias=True)

        # set weights to follow xavier
        for m in self.modules():
            weight_init(m)

    def forward(self, z, y):
        concated = torch.cat([z, y], 1)
        l1 = self.fc1(concated)
        l1 = torch_func.leaky_relu(l1, negative_slope=self.alpha)

        l2 = self.fc2(l1)
        out = torch_func.tanh(l2)

        return out

# define Discriminator
class Discriminator(torch.nn.Module):
    def __init__(self, x_size, y_size, n_hidden=128, n_out=1, alpha=0.2):
        super().__init__()
        self.alpha = alpha

        # define 2 fully connected layers
        input_size = x_size + y_size
        self.fc1 = torch.nn.Linear(input_size, n_hidden, bias=True)
        self.fc2 = torch.nn.Linear(n_hidden, n_out, bias=True)

        # set weights to follow xavier
        for m in self.modules():
            weight_init(m)

    def forward(self, x, y):
        concated = torch.cat([x, y], 1)
        l1 = self.fc1(concated)
        l1 = torch_func.leaky_relu(l1, negative_slope=self.alpha)

        l2 = self.fc2(l1)
        out = torch_func.sigmoid(l2)

        return out

class CGAN(object):
    def __init__(self, x_size, y_size, z_size, learning_rate):
        self.x_size, self.y_size, self.z_size = x_size, y_size, z_size

        # create CGAN network
        self.G = Generator(z_size, y_size).cuda()
        self.D = Discriminator(x_size, y_size).cuda()

        # optimizer
        beta1 = 0.5
        self.bce_loss = torch.nn.BCELoss().cuda()
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=learning_rate, betas=[beta1, 0.999])
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=learning_rate, betas=[beta1, 0.999])


# image save function
def save_generator_output(G, fixed_z, img_str, title):
    n_images = fixed_z.size()[0]
    n_rows = np.sqrt(n_images).astype(np.int32)
    n_cols = np.sqrt(n_images).astype(np.int32)

    z_ = Variable(fixed_z.cuda())
    samples = G(z_)
    samples = samples.cpu().data.numpy()

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5, 5), sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
    plt.savefig(img_str)
    plt.close(fig)

def train(net, data_loc, batch_size, epochs, print_every=30):
    # data_loader normalize [0, 1] ==> [-1, 1]
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_loc, train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)

    step = 0
    losses = []
    fixed_z = torch.Tensor(25, net.z_size).uniform_(-1, 1).cuda()

    for e in range(epochs):
        for x_, _ in train_loader:
            step += 1
            '''
            Train in Discriminator
            '''
            # reshape input image
            x_ = x_.view(-1, net.x_size)
            current_batch_size = x_.size()[0]

            # create labels for loss computation
            y_real_ = torch.ones(current_batch_size)
            y_fake_ = torch.zeros(current_batch_size)

            # make it cuda Tensor
            x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())

            # run real input on Discriminator
            D_result_real = net.D(x_)
            D_loss_real = net.BCE_loss(D_result_real, y_real_)

            # run Generator input on Discriminator
            z1_ = torch.Tensor(current_batch_size, net.z_size).uniform_(-1, 1)
            z1_ = Variable(z1_.cuda())
            x_fake = net.G(z1_)
            D_result_fake = net.D(x_fake)
            D_loss_fake = net.BCE_loss(D_result_fake, y_fake_)

            D_loss = D_loss_real + D_loss_fake

            # optimize Discriminator
            net.D.zero_grad()
            D_loss.backward()
            net.D_opt.step()

            '''
            Train in Generator
            '''
            z2_ = torch.Tensor(current_batch_size, net.z_size).uniform_(-1, 1)
            y_ = torch.ones(current_batch_size)
            z2_, y_ = Variable(z2_.cuda()), Variable(y_.cuda())
            G_result = net.G(z2_)
            D_result_fake2 = net.D(G_result)
            G_loss = net.BCE_loss(D_result_fake2, y_)

            net.G.zero_grad()
            G_loss.backward()
            net.G_opt.step()

            if step % print_every == 0:
                losses.append((D_loss.data[0], G_loss.data[0]))

                print("Epoch {}/{}...".format(e + 1, epochs),
                      "Discriminator Loss: {:.4f}...".format(D_loss.data[0]),
                      "Generator Loss: {:.4f}".format(G_loss.data[0]))

                # Sample from generator as we're training for viewing afterwards
        image_fn = './assets/epoch_{:d}_pytorch.png'.format(e)
        image_title = 'epoch {:d}'.format(e)
        save_generator_output(net.G, fixed_z, image_fn, image_title)

    return losses


def main():
    assets_dir = './assets/'
    if not os.path.isdir(assets_dir):
        os.mkdir(assets_dir)

    # parameters
    mnist_data_set = '../data_set/MNIST_data'
    x_size = 28 * 28
    y_size = 10
    z_size = 100
    learning_rate = 0.001
    batch_size = 64
    epochs = 30

    # build network
    net = CGAN(x_size, y_size, z_size, learning_rate)

    # training
    start_time = time.time()
    losses = train(net, mnist_data_set, batch_size, epochs)
    end_time = time.time()
    total_time = end_time - start_time
    print('Elapsed time: ', total_time)

    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('./assets/losses_pytorch.png')

    # create animated gif from result images
    images = []
    for e in range(epochs):
        image_fn = './assets/epoch_{:d}_pytorch.png'.format(e)
        images.append(imageio.imread(image_fn))
    imageio.mimsave('./assets/by_epochs_pytorch.gif', images, fps=3)


if __name__ == '__main__':
    main()
