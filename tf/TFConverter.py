import argparse

from backend.utils import load_weight
from model.slim_320 import create_slim_net

parser = argparse.ArgumentParser(
    description='convert model')

parser.add_argument('--pytorch_model', default=None, type=str)
parser.add_argument('--postprocess', action="store_true")
args = parser.parse_args()


def main():
    # input_shape = (240, 320)  # H,W
    input_shape = (96, 128)
    base_channel = 8 * 2
    num_classes = 2

    torch_path = args.pytorch_model or "/home/trdung/Documents/BoschPrj/lightFaceDetectModel/models/pretrained/version-slim-320.pth"
    mapping_table = "mapping_tables/slim_320.json"
    model = create_slim_net(input_shape, base_channel, num_classes, post_processing=args.postprocess)

    load_weight(model, torch_path, mapping_table)
    model.save(f'export_models/{args.net_type}.keras', include_optimizer=False)


if __name__ == '__main__':
    main()