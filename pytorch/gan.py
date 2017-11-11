import torch
import torch.nn as nn
import torch.nn.functional as functional


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform(m.weight.data)
        # torch.nn.init.constant(m.bias.data, 0)
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)
    return


class Generator(nn.Module):
    def __init__(self, z_size, y_size=0):
        super(Generator, self).__init__()
        self.im_size = 28
        self.n_filter = 512
        self.n_kernel = 4
        self.alpha = 0.2
        self.inputs_dim = z_size + y_size

        # make 3x3x512
        self.l1 = torch.nn.Linear(in_features=self.inputs_dim, out_features=3 * 3 * self.n_filter)

        # make 7x7x256
        self.l2 = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=self.n_filter, out_channels=self.n_filter // 2,
                                     kernel_size=self.n_kernel, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(num_features=self.n_filter // 2),
            nn.LeakyReLU(self.alpha)
        )

        # make 14x14x128
        self.l3 = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=self.n_filter // 2, out_channels=self.n_filter // 4,
                                     kernel_size=self.n_kernel, stride=2, padding=1, output_padding=0),
            torch.nn.BatchNorm2d(num_features=self.n_filter // 4),
            nn.LeakyReLU(self.alpha)
        )

        # make 28x28x1
        self.l4 = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=self.n_filter // 4, out_channels=1,
                                     kernel_size=self.n_kernel, stride=2, padding=1, output_padding=0)
        )

        # initialize weights with xavier_initialization
        for m in self.modules():
            weight_init(m)

    def forward(self, z, y=None):
        # concatenate inputs
        # z: [batch size, 100], y: [batch size, 10]
        if y is not None:
            inputs = torch.cat((z, y), dim=1)
        else:
            inputs = z

        # 1. reshape z-vector to fit as 2d shape image with fully connected layer
        l1 = self.l1(inputs)
        l1 = l1.view(-1, self.n_filter, 3, 3)
        l1 = functional.leaky_relu(l1, self.alpha)

        # 2. layer2 - [batch size, 512, 3, 3] ==> [batch size, 256, 7, 7]
        l2 = self.l2(l1)

        # 3. layer3 - [batch size, 256, 7, 7] ==> [batch size, 128, 14, 14]
        l3 = self.l3(l2)

        # 4. layer4 - [batch size, 128, 14, 14] ==> [batch size, 1, 28, 28]
        l4 = self.l4(l3)



