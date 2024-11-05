import platform
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default="export_models/128.keras", type=str,
                    help='Path to folder of exported save model')
parser.add_argument('--out', default="export_models/128_post_quant.tflite", type=str)
parser.add_argument('--repr_data', default='/home/trdung/Documents/BoschPrj/00_EDABK_Face_labels/VOCFormat/JPEGImages',
                    type=str, help='Path to folder of data for quantization calibration')
args = parser.parse_args()

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def preprocess_image(img):
    img_resize = cv2.resize(img, (128, 96))
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_resize = img_resize - 127.0
    img_resize = img_resize / 128.0
    img_resize = np.float32(np.expand_dims(img_resize, axis=0))

    return img_resize


def representative_dataset_generator():
    folder = Path(args.repr_data)

    i = 0
    for p in folder.iterdir():
        if p.is_dir():
            continue

        if i > 100:
            break

        img = cv2.imread(str(p))
        i += 1
        yield [preprocess_image(img)]


def main():
    model = tf.keras.models.load_model(args.save_dir)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    print("Type of model:", type(args.save_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.repr_data is None:
        raise Exception("repr_data must be provided to fully quantize the model")
    converter.representative_dataset = representative_dataset_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    open(args.out, "wb").write(tflite_model)


def test():
    interpreter = tf.lite.Interpreter(model_path=args.out)

    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()

    float_tensors = []
    for tensor in tensor_details:
        print(tensor)
        # if tensor['dtype'] == np.int32:
        #     float_tensors.append(tensor['name'])

    # if float_tensors:
    #     print("The model contains int32 tensors:")
    #     for name in float_tensors:
    #         print(name)
    # else:
    #     print("All tensors are quantized to int8/uint8.")

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.imread('imgs/test_input.jpg')
    img_resize = preprocess_image(img)

    # Convert to int8 input tensors if model is quantized
    params = input_details[0]["quantization_parameters"]
    if params["scales"] and params["zero_points"]:
        img_resize = (img_resize / params["scales"][0] + params["zero_points"][0]).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], img_resize)

    # first 3 times is warmup
    for _ in range(3):
        interpreter.invoke()
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    print(results)


if __name__ == '__main__':
    # main()
    test()