import json
from pprint import pprint

import utils
import gan
import cgan
import wgan
import wgan_gp


def main():
    # get training parameters
    with open('params.json') as f:
        gan_params = json.load(f)

    print('--params--')
    pprint(gan_params)

    dataset_base_dir = './data_set'
    for param in gan_params:
        model_name = param["model-name"]
        epochs = int(param["epochs"])
        mnist_type = param["mnist-type"]
        mnist = utils.get_mnist(dataset_base_dir, mnist_type)

        print('Training {:s} with epochs: {:d}, dataset: {:s}'.format(model_name, epochs, mnist_type))
        net = None

        if model_name == 'gan':
            net = gan.GAN(model_name, mnist_type, mnist, epochs)
        elif model_name == 'cgan':
            net = cgan.CGAN(model_name, mnist_type, mnist, epochs)
        elif model_name == 'wgan':
            net = wgan.WGAN(model_name, mnist_type, mnist, epochs)
        elif model_name == 'wgan-gp':
            net = wgan_gp.WGANGP(model_name, mnist_type, mnist, epochs)
        else:
            net = None

        if net is None:
            raise ValueError('Unable to recognize model type of {:s}'.format(model_name))

        net.train()

    return


if __name__ == '__main__':
    main()