import argparse
import sys
import cv2
import numpy
import psutil
import torch
# import torch.nn as nn
# import torch.quantization
#
# from pytorch.ssd.config.fd_config import define_img_size
# from pytorch.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor, fuse_model
# input_img_size = 128
# define_img_size(input_img_size)  # must put define_img_size() before importing create_mb_tiny_fd
#
# label_path = "ckpt/voc-model-labels.txt"
#
# class_names = [name.strip() for name in open(label_path).readlines()]
# num_classes = len(class_names)
# test_device = 'cpu'
#
# candidate_size = 20
# model_path = "ckpt/pretrained/version-slim-320.pth"
#
# # Create model
# net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
#
# # Set model to evaluation mode
# net.eval()
#
# # Load model weights before quantization
# net.load_state_dict(torch.load(model_path, map_location=torch.device(test_device)))
# # Since we have a quantized model, we need to provide an example input for tracing
# example_input = torch.randn(1, 3, 96, 128)
#
# # Convert the quantized model to TorchScript
# scripted_model = torch.jit.trace(net, example_input)
#
# # Save the scripted model
# scripted_model.save('ckpt/pretrained/128_model.pt')

model = torch.jit.load('ckpt/pretrained/128_model.pt')
model.eval()

input = torch.randn(1,3,96,128)
output = model(input)

print(model)
print(f"Input shape: {input.shape}")
print(f"Output shape: {output.shape}")

# for i, out in enumerate(output):
#     print(f"Output {i} shape: {out.shape}")
#
# out = torch.cat((output[0], output[1]), dim=-1)
# print(f"Output shape: {out.shape}")