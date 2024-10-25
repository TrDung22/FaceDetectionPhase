from functools import partial
import cv2
import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, filepath, inputSize=(128, 96), confThreshold=0.5,
                 centerVariance=0.1, sizeVariance=0.2,
                 nmsOutputSizeMAX=200, iouThreshold=0.3) -> None:

        # self.featureMaps = np.array([[40, 30], [20, 15], [10, 8], [5, 4]]) #320x240
        self.featureMaps = np.array([[16, 12],[8, 6],[4, 3],[2, 2]]) #128x96
        self.boxesMIN = np.array([[10, 16, 24], [32, 48, 0],
                                    [64, 96, 0], [128, 192, 256]])

        self.resize = partial(cv2.resize, dsize=inputSize)
        self.inputSize = np.array(inputSize)[:, None]

        self.anchorsXY, self.anchorsWH = self.anchorsGenerator()

        anchorsTotal = sum([fm[0] * fm[1] * len(mb) for fm, mb in zip(self.featureMaps, self.boxesMIN)])
        print(f"Total anchors: {anchorsTotal}")

        self.confThreshold = confThreshold
        self.centerVariance = centerVariance
        self.sizeVariance = sizeVariance
        self.nms = partial(tf.image.non_max_suppression,
                            max_output_size=nmsOutputSizeMAX,
                            iou_threshold=iouThreshold)

        # tflite model init
        self.interpreter = tf.lite.Interpreter(model_path=filepath)
        self.interpreter.allocate_tensors()

        # model details
        inputDetails = self.interpreter.get_input_details()
        outputDetails = self.interpreter.get_output_details()

        # inference helper
        self.inputTensorSetter = partial(self.interpreter.set_tensor,
                                         inputDetails[0]["index"])
        self.boxesTensorGetter = partial(self.interpreter.get_tensor,
                                         outputDetails[0]["index"])
        self.scoresTensorGetter = partial(self.interpreter.get_tensor,
                                          outputDetails[1]["index"])

    def anchorsGenerator(self):
        anchors = []
        for featureMapWH, boxMIN in zip(self.featureMaps, self.boxesMIN):
            boxMIN = boxMIN[boxMIN > 0]
            if len(boxMIN) == 0:
                continue

            gridWH = boxMIN / self.inputSize
            gridWH = np.tile(gridWH.T, (np.prod(featureMapWH), 1))

            gridXY = np.meshgrid(range(featureMapWH[0]),
                                  range(featureMapWH[1]))
            gridXY = np.add(gridXY, 0.5)

            gridXY /= featureMapWH[..., None, None]

            gridXY = np.stack(gridXY, axis=-1)
            gridXY = np.tile(gridXY, [1, 1, len(boxMIN)])
            gridXY = gridXY.reshape(-1, 2)

            prior = np.concatenate((gridXY, gridWH), axis=-1)
            anchors.append(prior)

        anchors = np.concatenate(anchors, axis=0)
        anchors = np.clip(anchors, 0.0, 1.0)

        return anchors[:, :2], anchors[:, 2:]

    def preProcessing(self, img):
        resized = self.resize(img)
        imageRGB = resized[..., ::-1]
        imageNorm = imageRGB.astype(np.float32)
        cv2.normalize(imageNorm, imageNorm,
                      alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)
        return imageNorm[None, ...]

    def inference(self, img):
        # BGR image to tensor
        inputTensor = self.preProcessing(img)

        # set tensor and invoke
        self.inputTensorSetter(inputTensor)
        self.interpreter.invoke()

        # get results
        boxes = self.boxesTensorGetter()[0]
        scores = self.scoresTensorGetter()[0]

        # decode boxes to corner format
        boxes, scores = self.postProcessing(boxes, scores)
        boxes *= np.tile(img.shape[1::-1], 2)

        return boxes, scores

    def postProcessing(self, boxes, scores):
        # bounding box regression
        boxes = self.decodeRegression(boxes)
        scores = scores[:, 1]

        # confidence threshold filter
        maskConf = self.confThreshold < scores
        boxes, scores = boxes[maskConf], scores[maskConf]

        # non-maximum suppression
        nmsMask = self.nms(boxes=boxes, scores=scores)
        boxes = np.take(boxes, nmsMask, axis=0)

        return boxes, scores

    def decodeRegression(self, reg):
        # bounding box regression
        centerXY = reg[:, :2] * self.centerVariance * \
            self.anchorsWH + self.anchorsXY
        centerWH = np.exp(
            reg[:, 2:] * self.sizeVariance) * self.anchorsWH / 2

        # center to corner
        startXY = centerXY - centerWH
        endXY = centerXY + centerWH

        boxes = np.concatenate((startXY, endXY), axis=-1)
        boxes = np.clip(boxes, 0.0, 1.0)

        return boxes
