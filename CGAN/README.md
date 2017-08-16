# Conditional GAN on MNIST - tensorflow

## Reference
http://wiseodd.github.io/techblog/2016/12/24/conditional-gan-tensorflow/

## Network at a glance

| **Generator**, **Discriminator** |
| --- |
| ![N](./assets/network_structure.png) |

### Training Losses

| tensorflow |
| --- |
| ![](./assets/losses_tf.png) |

### Generated samples via epochs

| epochs | tensorflow |
| --- | --- |
| 0 | ![](./assets/epoch_0_tf.png) |
| 9 | ![](./assets/epoch_9_tf.png) |
| 19 | ![](./assets/epoch_19_tf.png) |
| 29 | ![](./assets/epoch_29_tf.png) |
|  | ![](./assets/by_epochs_tf.gif) |

### Generated results by epochs - fixed classes
| tensorflow |
| --- |
| ![](./assets/cGAN-MNIST-result-by-epoch.PNG) |

### Generated results by epochs - fixed classes & fixed z
| tensorflow |
| --- |
| ![](./assets/cGAN-MNIST-result-by-style.PNG) |
