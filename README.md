# Get Started

## **Environment**
- python >= 3.8.0
- pytorch >= 1.8
- CUDA 11.8+

## **Installation **

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
```

c. Install required packages

```shell
git clone https://github.com/JaebinLimm/YOLOv5-Socket.git
cd YOLOv5-Socket
pip install -r requirements.txt
```

## **Quick Start**
The pre-trained model can be found [here](https://github.com/ultralytics/yolov5). You can test the model by running the command below.

```shell
python socket_server.py
python client.py
# Please run socket_server.py and client.py in separate terminal windows.
# Run socket_server.py first, then client.py.
```

If you have any questions, please feel free to contact the writer at rkfakehd112@gmail.com.

## **References**

I referenced the repo below for the code.
- [YOLOv5](https://github.com/ultralytics/yolov5).

