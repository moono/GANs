# DCGAN on celebA dataset - tensorflow & pytorch

## Network at a glance

| **Generator**, **Discriminator** |
| --- |
| ![N](./assets/network_structure.png) |

### Training Losses

| tensorflow | pytorch |
| --- | --- |
| ![](./assets/losses_tf.png) | ![](./assets/losses_pytorch.png) |

### Generated samples via epochs

| epochs | tensorflow | pytorch |
| --- | --- | --- |
| 0 | ![](./assets/epoch_0_tf.png) | ![](./assets/epoch_0_pytorch.png) |
| 7 | ![](./assets/epoch_7_tf.png) | ![](./assets/epoch_7_pytorch.png) |
| 15 | ![](./assets/epoch_15_tf.png) | ![](./assets/epoch_15_pytorch.png) |
| 19 | ![](./assets/epoch_19_tf.png) | ![](./assets/epoch_19_pytorch.png) |
|  | ![](./assets/by_epochs_tf.gif) | ![](./assets/by_epochs_pytorch.gif) |