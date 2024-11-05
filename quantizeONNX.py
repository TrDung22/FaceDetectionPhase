import torch.onnx
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import glob
from PIL import Image
import numpy as np

# 2. Chuẩn Bị Dữ Liệu Hiệu Chuẩn
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 96))
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose(2, 0, 1)
    image_data = image_data / 255.0
    image_data = np.expand_dims(image_data, axis=0)
    return image_data

calibration_dataset = []
image_paths = glob.glob('/home/trdung/Documents/BoschPrj/00_EDABK_Face_labels/CalibrateData/*.jpg')
for image_path in image_paths:
    image_data = preprocess_image(image_path)
    calibration_dataset.append(image_data)

# 3. Định Nghĩa Data Reader cho Hiệu Chuẩn
class CalibDataReader(CalibrationDataReader):
    def __init__(self, calibration_dataset):
        self.data = calibration_dataset
        self.iterator = iter(self.data)

    def get_next(self):
        try:
            input_data = next(self.iterator)
            return {'input': input_data}
        except StopIteration:
            return None

calibration_data_reader = CalibDataReader(calibration_dataset)

# 4. Thực Hiện Quantization Tĩnh
model_path = 'ckpt/onnx/128_model.onnx'
quantized_model_path = 'ckpt/onnxQuantized/128_quantized.onnx'

quantize_static(
    model_input=model_path,
    model_output=quantized_model_path,
    calibration_data_reader=calibration_data_reader
)

# 5. Kiểm Tra Mô Hình Đã Quantize
quantized_model = onnx.load(quantized_model_path)
onnx.checker.check_model(quantized_model)

ort_session = ort.InferenceSession(quantized_model_path)

# Chạy suy luận trên dữ liệu kiểm tra
test_input = preprocess_image('tf/imgs/test_input.jpg')
outputs = ort_session.run(None, {'input': test_input})
