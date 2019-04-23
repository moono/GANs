import json
from pprint import pprint
from importlib import import_module


def main():
    # get training parameters
    with open('params.json') as f:
        gan_params = json.load(f)

    print('--params--')
    pprint(gan_params)

    for param in gan_params:
        model_name = param["model-name"]
        mnist_type = param["mnist-type"]
        gan_loss_type = param["gan_loss_type"]

        print('Training {:s} on dataset: {:s}'.format(model_name, mnist_type))

        # get appropriate module and it's class to start training
        module_name = import_module(model_name)
        gan_class = getattr(module_name, model_name.upper())
        net = gan_class(model_name, mnist_type, gan_loss_type)
        net.train()
    return


if __name__ == '__main__':
    main()
