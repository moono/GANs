# GANs
Study Generative Adversarial Networks

## Prerequisites

* Conda environment in Windows 10 & Ubuntu 14.04 **(Currently pytorch is only tested on Ubuntu)**
* Key components
  * python=3.5
  * tensorflow=1.2.1
  * pytorch=0.1.12

```shell
# Windows
conda create -n tf-pytorch python=3.5
activate tf-pytorch
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.2.1-cp35-cp35m-win_amd64.whl
conda install -c peterjc123 pytorch
```

```shell
# Ubuntu
conda create -n tf-pytorch python=3.5
source activate tf-pytorch
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.2.1-cp35-cp35m-win_amd64.whl
conda install pytorch torchvision cuda80 -c soumith
```

## Acknowledgments

* 
