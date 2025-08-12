import cv2
import os
import socketio
import base64
import yolo_detect
from utils.dataloaders import LoadStreams
from opt import parse_opt
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import increment_path
from pathlib import Path

sio = socketio.Client()
opt = parse_opt()

device = select_device(opt.device)

model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn, data=opt.data, fp16=opt.half)
cam = LoadStreams(['0'][0], img_size=opt.imgsz, stride=32, auto=bool("True"), vid_stride=opt.vid_stride)

save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
(save_dir / "labels" if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)
csv_path = save_dir / "predictions.csv"
img_path = save_dir / "img"

os.makedirs(img_path, exist_ok=True)

@sio.event
def connect():
    print("Connected to the server")

@sio.event
def disconnect():
    print("Disconnected from the server.")

sio.connect('http://localhost:5000')

def send_detection(image, label, timestamp):

    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    sio.emit('detection_result', {
        'image': img_base64,
        'label': label,
        'time': timestamp
    })

while True:
    yolo_result = yolo_detect.main(opt, cam, model, save_dir, csv_path)
    time = str(yolo_result[1])
    label = str(yolo_result[2])
    send_detection(yolo_result[0], time, label)
