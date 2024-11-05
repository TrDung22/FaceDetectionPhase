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
input_img_size = 128
define_img_size(input_img_size)  # must put define_img_size() before importing create_mb_tiny_fd

label_path = "ckpt/voc-model-labels.txt"

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = 'cpu'

candidate_size = 20
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
torch.ao.quantization.convert(net, inplace=True)

# Load model weights before quantization
net.load_state_dict(torch.load(model_path, map_location=torch.device(test_device)))
# Since we have a quantized model, we need to provide an example input for tracing
example_input = torch.randn(1, 3, 96, 128)

# Convert the quantized model to TorchScript
scripted_model = torch.jit.trace(net, example_input)

# Save the scripted model
scripted_model.save('ckpt/pretrained/128_post_quant.pt')
