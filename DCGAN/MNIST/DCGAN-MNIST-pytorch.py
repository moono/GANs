import torch
from torch.nn import init
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt 
import numpy as np
import os
import time
import imageio

def my_weight_init(m):
    if isinstance(m, torch.nn.Linear):
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0)

class Generator(torch.nn.Module):
    def __init__(self, z_size, initial_feature_size=512, n_output_channel=1, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.initial_feature_size = initial_feature_size

        # make 3x3x512
        n_first_layer_units = 3 * 3 * initial_feature_size
        self.fc1 = torch.nn.Linear(z_size, n_first_layer_units, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(initial_feature_size)

        # make 7x7x256
        self.deconv2 = torch.nn.ConvTranspose2d(initial_feature_size, initial_feature_size//2, kernel_size=3, stride=2, padding=0, bias=True, output_padding=0)
        self.bn2 = torch.nn.BatchNorm2d(initial_feature_size//2)

        # make 14x14x128
        self.deconv3 = torch.nn.ConvTranspose2d(initial_feature_size//2, initial_feature_size//4, kernel_size=5, stride=2, padding=2, bias=True, output_padding=1)
        self.bn3 = torch.nn.BatchNorm2d(initial_feature_size//4)

        # make 28x28x1
        self.deconv4 = torch.nn.ConvTranspose2d(initial_feature_size//4, n_output_channel, kernel_size=5, stride=2, padding=2, bias=True, output_padding=1)

        for m in self.modules():
            my_weight_init(m)
            

    def forward(self, input):
        l1 = self.fc1(input)
        l1 = l1.view(-1, self.initial_feature_size, 3, 3) # reshape
        l1 = torch.nn.functional.leaky_relu(self.bn1(l1), negative_slope=self.alpha)

        l2 = torch.nn.functional.leaky_relu(self.bn2(self.deconv2(l1)), negative_slope=self.alpha)
        l3 = torch.nn.functional.leaky_relu(self.bn3(self.deconv3(l2)), negative_slope=self.alpha)
        l4 = self.deconv4(l3)
        out = torch.nn.functional.tanh(l4)

        return out

class Discriminator(torch.nn.Module):
    def __init__(self, x_size, initial_feature_size=64, n_output=1, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.initial_feature_size = initial_feature_size

        # input is 28x28x1

        # make 14x14x64
        self.conv1 = torch.nn.Conv2d(x_size, initial_feature_size, kernel_size=5, stride=2, padding=2, bias=True)

        # make 7x7x128
        self.conv2 = torch.nn.Conv2d(initial_feature_size, initial_feature_size*2, kernel_size=5, stride=2, padding=2, bias=True)
        self.bn2 = torch.nn.BatchNorm2d(initial_feature_size*2)

        # make 4x4x256
        self.conv3 = torch.nn.Conv2d(initial_feature_size*2, initial_feature_size*4, kernel_size=5, stride=2, padding=2, bias=True)
        self.bn3 = torch.nn.BatchNorm2d(initial_feature_size*4)

        self.fc4 = torch.nn.Linear(4 * 4 * initial_feature_size*4, n_output, bias=True)

        for m in self.modules():
            my_weight_init(m)

    def forward(self, input):
        l1 = torch.nn.functional.leaky_relu(self.conv1(input), negative_slope=self.alpha)
        l2 = torch.nn.functional.leaky_relu(self.bn2(self.conv2(l1)), negative_slope=self.alpha)
        l3 = torch.nn.functional.leaky_relu(self.bn3(self.conv3(l2)), negative_slope=self.alpha)
        flattened = l3.view(-1, 4 * 4 * self.initial_feature_size*4) # reshape
        l4 = self.fc4(flattened)
        out = torch.nn.functional.sigmoid(l4)

        return out

# image save function
def save_generator_output(G, fixed_z, img_str, title):
    n_images = fixed_z.size()[0]
    n_rows = np.sqrt(n_images).astype(np.int32)
    n_cols = np.sqrt(n_images).astype(np.int32)
    
    z_ = Variable(fixed_z.cuda())
    samples = G(z_)
    samples = samples.cpu().data.numpy()

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
Parameters
'''
image_width = 28
image_height = 28
image_channels = 1
x_size = image_channels
z_size = 100
# n_hidden = 128
# n_classes = 10
epochs = 30
batch_size = 64
learning_rate = 0.0002
alpha = 0.2
beta1 = 0.5
print_every = 50

# data_loader normalize [0, 1] ==> [-1, 1]
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../../data_set/MNIST_data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# build network
G = Generator(z_size, n_output_channel=image_channels, alpha=alpha)
D = Discriminator(x_size, n_output=1, alpha=alpha)
G.cuda()
D.cuda()

# optimizer
BCE_loss = torch.nn.BCELoss()
G_opt = torch.optim.Adam( G.parameters(), lr=learning_rate, betas=[beta1, 0.999] )
D_opt = torch.optim.Adam( D.parameters(), lr=learning_rate, betas=[beta1, 0.999] )

assets_dir = './assets/'
if not os.path.isdir(assets_dir):
    os.mkdir(assets_dir)

'''
Start training
'''
step = 0
samples = []
losses = []
fixed_z = torch.Tensor(25, z_size).uniform_(-1, 1)
start_time = time.time()
for e in range(epochs):
    for x_, _ in train_loader:
        step += 1
        '''
        Train in Discriminator
        '''
        # reshape input image
        #x_ = x_.view(-1, image_channels, image_width, image_height)
        # print(x_.size())
        current_batch_size = x_.size()[0]

        # create labels for loss computation
        y_real_ = torch.ones(current_batch_size)
        y_fake_ = torch.zeros(current_batch_size)

        # make it cuda Tensor
        x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())

        # run real input on Discriminator
        D_result_real = D(x_)
        D_loss_real = BCE_loss(D_result_real, y_real_)

        # run Generator input on Discriminator
        z1_ = torch.Tensor(current_batch_size, z_size).uniform_(-1, 1)
        z1_ = Variable(z1_.cuda())
        x_fake = G(z1_)
        D_result_fake = D(x_fake)
        D_loss_fake = BCE_loss(D_result_fake, y_fake_)

        D_loss = D_loss_real + D_loss_fake

        # optimize Discriminator
        D.zero_grad()
        D_loss.backward()
        D_opt.step()
        
        '''
        Train in Generator
        '''
        z2_ = torch.Tensor(current_batch_size, z_size).uniform_(-1, 1)
        y_ = torch.ones(current_batch_size)
        z2_, y_ = Variable(z2_.cuda()), Variable(y_.cuda())
        G_result = G(z2_)
        D_result_fake2 = D(G_result)
        G_loss = BCE_loss(D_result_fake2, y_)

        G.zero_grad()
        G_loss.backward()
        G_opt.step()

        if step % print_every == 0:
            losses.append((D_loss.data[0], G_loss.data[0]))

            print("Epoch {}/{}...".format(e+1, epochs),
                "Discriminator Loss: {:.4f}...".format(D_loss.data[0]),
                "Generator Loss: {:.4f}".format(G_loss.data[0])) 
    # Sample from generator as we're training for viewing afterwards
    image_fn = './assets/epoch_{:d}_pytorch.png'.format(e)
    image_title = 'epoch {:d}'.format(e)
    save_generator_output(G, fixed_z, image_fn, image_title)

end_time = time.time()
total_time = end_time - start_time
print('Elapsed time: ', total_time)
# 30 epochs: 751.90

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
    images.append( imageio.imread(image_fn) )
imageio.mimsave('./assets/by_epochs_pytorch.gif', images, fps=3)