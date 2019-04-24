# GAN
* Generative Adversarial Networks
  * GAN
  * CGAN
  * WGAN
  * WGAN-GP

## Results - MNIST
| Type | Generation | Loss |
| ---- | --------- | ---- |
| GAN | ![](./results/gan/mnist-v2-e030.png) | ![](./results/gan/mnist-v2-losses.png) |
| WGAN | ![](./results/wgan/mnist-e030.png) | ![](./results/wgan/mnist-losses.png) |
| WGAN-GP | ![](./results/wgangp/mnist-e030.png) | ![](./results/wgangp/mnist-losses.png) |
| CGAN | ![](./results/cgan/mnist-v2-e030.png) | ![](./results/cgan/mnist-v2-losses.png) |
| ACGAN | ![](./results/acgan/mnist-v2-e030.png) | ![](./results/acgan/mnist-v2-losses.png) |


## Results - fashion-MNIST
| Type | Generation | Loss |
| ---- | --------- | ---- |
| GAN | ![](./results/gan/fashion_mnist-v2-e030.png) | ![](./results/gan/fashion_mnist-v2-losses.png) |
| WGAN | ![](./results/wgan/fashion_mnist-e030.png) | ![](./results/wgan/fashion_mnist-losses.png) |
| WGAN-GP | ![](./results/wgangp/fashion_mnist-e030.png) | ![](./results/wgangp/fashion_mnist-losses.png) |
| CGAN | ![](./results/cgan/fashion_mnist-v2-e030.png) | ![](./results/cgan/fashion_mnist-v2-losses.png) |
| ACGAN | ![](./results/acgan/fashion_mnist-v2-e030.png) | ![](./results/acgan/fashion_mnist-v2-losses.png) |

## Prerequisites
* Ubuntu 16.04
* tensorflow-gpu==1.13.1
* tensorflow-datasets==1.0.1

## Data sets
* MNIST & fashion-MNIST: `dataset_loader.py`

## Reference
* Based on: https://github.com/hwalsuklee/tensorflow-generative-model-collections
