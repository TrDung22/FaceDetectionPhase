import math
import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.quantization


sys.path.append('../')
from pytorch.ssd.config.fd_config import define_img_size
from pytorch.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor, fuse_model

# Định nghĩa kích thước đầu vào
input_img_size = 128
define_img_size(input_img_size)

# Các thông số khác
label_path = "../ckpt/voc-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = "cpu"
candidate_size = 800
threshold = 0.5
val_image_root = "/home/trdung/Documents/BoschPrj/00_EDABK_Face_labels/VOCFormat/JPEGImages"
val_result_txt_save_root = "./evaluationPytorch128Quantize/"
model_path = "../ckpt/pretrained/128_post_quant.pth"

# Tạo mô hình và chuẩn bị cho lượng tử hóa
net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
net.eval()
net = fuse_model(net)
net.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
torch.ao.quantization.prepare(net, inplace=True)
torch.quantization.convert(net, inplace=True)

# Tạo predictor
predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)

# Tải trọng lượng
net.load_state_dict(torch.load(model_path, map_location=torch.device(test_device)))

# Đoạn mã xử lý hình ảnh
counter = 0
for parent, dir_names, file_names in os.walk(val_image_root):
    for file_name in file_names:
        if not file_name.lower().endswith('jpg'):
            continue
        im = cv2.imread(os.path.join(parent, file_name), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(im, candidate_size / 2, threshold)

        event_name = parent.split('/')[-1]
        if not os.path.exists(os.path.join(val_result_txt_save_root, event_name)):
            os.makedirs(os.path.join(val_result_txt_save_root, event_name))
        with open(os.path.join(val_result_txt_save_root, event_name, file_name.split('.')[0] + '.txt'), 'w') as fout:
            fout.write(file_name.split('.')[0] + '\n')
            fout.write(str(boxes.size(0)) + '\n')
            for i in range(boxes.size(0)):
                bbox = boxes[i, :]
                fout.write('%d %d %d %d %.03f\n' % (
                    math.floor(bbox[0]),
                    math.floor(bbox[1]),
                    math.ceil(bbox[2] - bbox[0]),
                    math.ceil(bbox[3] - bbox[1]),
                    min(probs[i], 1.0)
                ))
        counter += 1
        print('[%d] %s is processed.' % (counter, file_name))
