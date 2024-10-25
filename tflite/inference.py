import argparse
import cv2
import time

from TFLiteModel import Model


parser = argparse.ArgumentParser(description='TFLite Face Detector')
parser.add_argument('--img_path', type=str, help='Image path for inference')
parser.add_argument('--video_path', type=str, help='Video path for inference')

args = parser.parse_args()


def imageInference(imagePath, modelPath, color=(125, 255, 0)):

    fd = Model(modelPath, confThreshold=0.5)

    img = cv2.imread(imagePath)

    boxes, scores = fd.inference(img)

    for result in boxes.astype(int):
        cv2.rectangle(img, (result[0], result[1]),
                      (result[2], result[3]), color, 2)

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def videoInference(video, modelPath, color=(125, 255, 0)):
    fd = Model(modelPath, confThreshold=0.5)

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(video)

    frameCounter = 0
    timeTotal = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        timeStart = time.perf_counter()
        boxes, scores = fd.inference(frame)
        timeInference = time.perf_counter() - timeStart
        timeTotal += timeInference
        frameCounter += 1

        # Tính FPS
        fps = frameCounter / timeTotal if timeTotal > 0 else 0

        print(f"Inference time: {timeInference:.3f}s, FPS: {fps:.2f}")

        for box, score in zip(boxes.astype(int), scores):
            cv2.rectangle(frame, (box[0], box[1]),
                          (box[2], box[3]), color, 2)
            # Hiển thị độ tin cậy lên trên hộp
            cv2.putText(frame, f'{score:.2f}', (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Hiển thị FPS trên khung hình
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('res', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ckptPath = "pretrained/128_post_quant.tflite"
    if args.img_path:
        imageInference(args.img_path, ckptPath)
    elif args.video_path:
        videoInference(args.video_path, ckptPath)
    else:
        print('--ima_path or --video_path must be filled')
