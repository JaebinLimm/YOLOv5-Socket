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
# CUDA 11.8
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

## **Acknowledgements**

This code is built upon YOLOv5.

