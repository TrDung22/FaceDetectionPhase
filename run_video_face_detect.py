import argparse
import sys
import cv2
import numpy
import psutil
import torch
import torch.nn as nn
import torch.quantization

from pytorch.ssd.config.fd_config import define_img_size
from pytorch.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor, fuse_model
from pytorch.utils.misc import Timer

parser = argparse.ArgumentParser(description='detect_video')

parser.add_argument('--input_size', default=128, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.7, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=20, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cpu", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--video_path', default="/home/trdung/Documents/BoschPrj/iBME_P-001_D1_H.mp4", type=str,
                    help='path of video')
parser.add_argument('--frame_interval', default=1, type=int, help='Process every nth frame')  # Added frame_interval
args = parser.parse_args()

input_img_size = args.input_size
define_img_size(input_img_size)  # must put define_img_size() before importing create_mb_tiny_fd

label_path = "ckpt/voc-model-labels.txt"

cap = cv2.VideoCapture(args.video_path)  # capture from video
# cap = cv2.VideoCapture(0)  # capture from camera

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = args.test_device

candidate_size = args.candidate_size
threshold = args.threshold
model_path = "ckpt/pretrained/128_post_quant.pth"

# Create model
net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)

# Set model to evaluation mode
net.eval()
# Fuse the model
net = fuse_model(net)

# Set quantization configuration
net.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')

# Prepare for quantization
torch.ao.quantization.prepare(net, inplace=True)

# Convert to quantized model
torch.quantization.convert(net, inplace=True)

# Create predictor
predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
# Load model weights
net.load_state_dict(torch.load(model_path, map_location=torch.device(test_device)))
# Initial
frame_count = 0
total_time = 0
timer = Timer()
last_boxes = None
last_probs = None

while True:

    ret, orig_image = cap.read()
    if orig_image is None:
        print("End of video or camera feed.")
        break

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    frame_count += 1
    timer.start()
    if frame_count % args.frame_interval == 0:
        boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
        # Update last_boxes, last_probs if there's a new detection
        if boxes.size(0) > 0:
            last_boxes = boxes.clone()
            last_probs = probs.clone()
        # Draw bounding boxes
        if boxes.size(0) > 0:
            for i in range(boxes.size(0)):
                box = boxes[i, :].numpy().astype(int)
                label = f"{probs[i]:.2f}"
                cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
                cv2.putText(orig_image, label,
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
    else:
        # Reuse bounding boxes from last frame
        if last_boxes is not None:
            for i in range(last_boxes.size(0)):
                box = last_boxes[i, :].numpy().astype(int)
                label = f"{last_probs[i]:.2f}"
                cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
                cv2.putText(orig_image, label,
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
    interval = timer.end()

    # Calculate FPS
    total_time += interval
    fps = frame_count / total_time
    cv2.putText(orig_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    orig_image = cv2.resize(orig_image, None, fx=0.8, fy=0.8)
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
