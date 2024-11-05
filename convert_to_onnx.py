"""
This code is used to convert the pytorch model into an onnx format model.
"""
import sys

import torch.onnx

from pytorch.ssd.config.fd_config import define_img_size

input_img_size = 128  # define input size ,default optional(128/160/320/480/640/1280)
define_img_size(input_img_size)
from pytorch.ssd.mb_tiny_fd import create_mb_tiny_fd, fuse_model

label_path = "ckpt/voc-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

model_path = "ckpt/pretrained/128_model.pt"
# model_path = "models/pretrained/version-slim-640.pth"
net = create_mb_tiny_fd(len(class_names), is_test=True, device="cpu")
net.eval()
net.load(model_path)

model_name = model_path.split("/")[-1].split(".")[0]
model_path = f"ckpt/onnx/128_model.onnx"

dummy_input = torch.randn(1, 3, 96, 128).to("cpu")
# dummy_input = torch.randn(1, 3, 480, 640).to("cuda") #if input size is 640*480
torch.onnx.export(net, dummy_input, model_path, verbose=False, input_names=['input'], output_names=['scores', 'boxes'])
