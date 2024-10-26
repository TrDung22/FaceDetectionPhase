import torch
import torch.nn as nn
from pytorch.ssd.config.fd_config import define_img_size
from pytorch.datasets.voc_dataset import VOCDataset
from torch.utils.data import DataLoader, Subset  # Import Subset
from pytorch.nn.multibox_loss import MultiboxLoss
define_img_size(128)

from pytorch.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor, fuse_model
from pytorch.ssd.data_preprocessing import TrainAugmentation, TestTransform
from pytorch.ssd.config import fd_config
from pytorch.ssd.ssd import MatchPrior

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

# Prepare model to be quantized
# Đặt lại tên biến cho rõ ràng hơn
batch_size = 5  # Số lượng mẫu mỗi batch
num_evaluate = 100  # Số lượng ví dụ muốn đánh giá

# Giả sử bạn đã tạo và tải trọng số cho mô hình SSD
model = create_mb_tiny_fd(num_classes=2, is_test=True, device='cpu')
model.load('ckpt/pretrained/version-slim-320.pth')
model.eval()

# Thực hiện fusion các lớp
model = fuse_model(model)

# Định nghĩa cấu hình quantization
model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
print(model.qconfig)

# Chuẩn bị mô hình cho quantization
torch.ao.quantization.prepare(model, inplace=True)

# Prepare data to be calibrated
DEVICE = 'cpu'
config = fd_config
criterion = MultiboxLoss(config.priors, neg_pos_ratio=3,
                         center_variance=0.1, size_variance=0.2, device=DEVICE)
print(fd_config.image_size)
test_transform = TestTransform(config.image_size, config.image_mean_test, config.image_std)
target_transform = MatchPrior(config.priors, config.center_variance,
                              config.size_variance, iou_threshold=0.35)

validation_dataset_path = '/home/trdung/Documents/BoschPrj/00_EDABK_Face_labels/VOCFormat'
val_dataset = VOCDataset(validation_dataset_path, transform=test_transform,
                         target_transform=target_transform, is_test=True)

# Tạo Subset với 100 ví dụ đầu tiên
subset_indices = list(range(num_evaluate))
subset_val_dataset = Subset(val_dataset, subset_indices)

# Tạo DataLoader cho Subset
val_subset_loader = DataLoader(
    subset_val_dataset,
    batch_size=batch_size,
    num_workers=4,
    shuffle=False
)

# Thực hiện đánh giá và calibration trên 100 ví dụ
test(val_subset_loader, model, criterion, DEVICE)

# Convert và lưu mô hình đã quantize
torch.ao.quantization.convert(model, inplace=True)
torch.save(model.state_dict(), 'ckpt/pretrained/128_post_quant.pth')
