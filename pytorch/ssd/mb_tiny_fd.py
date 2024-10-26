from torch.nn import Conv2d, Sequential, ModuleList, ReLU
import torch.nn as nn
from torch.quantization import fuse_modules
from pytorch.nn.mb_tiny import Mb_Tiny
from pytorch.ssd.config import fd_config as config
from pytorch.ssd.predictor import Predictor
from pytorch.ssd.ssd import SSD


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_mb_tiny_fd(num_classes, is_test=False, device="cuda"):
    base_net = Mb_Tiny(2)
    base_net_model = base_net.model  # disable dropout layer

    source_layer_indexes = [
        8,
        11,
        13
    ]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=base_net.base_channel * 16, out_channels=base_net.base_channel * 4, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=base_net.base_channel * 16, kernel_size=3, stride=2, padding=1),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=3 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=2 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=2 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=base_net.base_channel * 16, out_channels=3 * 4, kernel_size=3, padding=1)
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=3 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=2 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=base_net.base_channel * 16, out_channels=3 * num_classes, kernel_size=3, padding=1)
    ])

    return SSD(num_classes, base_net_model, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config, device=device)


def create_mb_tiny_fd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean_test,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor

def fuse_model(model):
    """
    Fuses Conv2d, BatchNorm2d và ReLU modules trong mô hình SSD để chuẩn bị cho post-training quantization.

    Args:
        model (nn.Module): Mô hình SSD cần thực hiện fusion.

    Returns:
        nn.Module: Mô hình sau khi đã thực hiện fusion các lớp.
    """
    # Fusing layers trong base_net_model
    for module in model.base_net:
        if isinstance(module, nn.Sequential):
            if len(module) == 3:
                # conv_bn block: Conv2d + BatchNorm2d + ReLU
                fuse_modules(module, ['0', '1', '2'], inplace=True)
            elif len(module) == 6:
                # conv_dw block: Conv2d + BatchNorm2d + ReLU + Conv2d + BatchNorm2d + ReLU
                fuse_modules(module, ['0', '1', '2'], inplace=True)
                fuse_modules(module, ['3', '4', '5'], inplace=True)

    # Fusing layers trong extras
    for extra in model.extras:
        if isinstance(extra, nn.Sequential):
            # SeperableConv2d: Conv2d + ReLU + Conv2d
            # Chỉ fuse Conv2d và ReLU đầu tiên
            if len(extra) >= 2:
                fuse_modules(extra, ['0', '1'], inplace=True)

    # Fusing layers trong classification_headers
    for header in model.classification_headers:
        if isinstance(header, nn.Sequential):
            # SeperableConv2d: Conv2d + ReLU + Conv2d
            # Chỉ fuse Conv2d và ReLU đầu tiên
            if len(header) >= 2:
                fuse_modules(header, ['0', '1'], inplace=True)

    # Fusing layers trong regression_headers
    for header in model.regression_headers:
        if isinstance(header, nn.Sequential):
            # SeperableConv2d: Conv2d + ReLU + Conv2d hoặc chỉ Conv2d + ReLU
            # Chỉ fuse Conv2d và ReLU đầu tiên nếu có
            if len(header) >= 2:
                if isinstance(header[1], nn.ReLU):
                    fuse_modules(header, ['0', '1'], inplace=True)

    return model