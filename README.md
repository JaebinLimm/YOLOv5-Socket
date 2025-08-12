# Get Started

## **Environment**
- python >= 3.8.0
- pytorch >= 1.8
- CUDA 10.2+

## **Install**

a. Create a conda virtual environment and activate it.

```shell
conda create -n YOLOv5-Socket python=3.8 -y
conda activate YOLOv5-Socket
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
# CUDA 10.2
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 –f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.3
- pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

# **Acknowledgements**

This code is built upon YOLOv5.

