"""
This code is used to evaluate TFlite implemented model on test dataset
"""
import os
import cv2
import math
from tflite.TFLiteModel import Model

def main():
    modelPath = '/home/trdung/Documents/BoschPrj/lightFaceDetectModel/tflite/pretrained/128_post_quant.tflite'
    testDir = '/home/trdung/Documents/BoschPrj/00_EDABK_Face_labels/VOCFormat/JPEGImages'
    outBboxDir = './evaluationTflite128postQuant/'
    confThreshold = 0.5

    # Initialize the face detector
    fd = Model(modelPath, input_size=(128,96), confThreshold=confThreshold)

    counter = 0

    # Walk through the validation image directory
    for parent, dirNames, fileNames in os.walk(testDir):
        for fileName in fileNames:
            if not fileName.lower().endswith(('.jpg')):
                continue
            imagePath = os.path.join(parent, fileName)
            img = cv2.imread(imagePath)
            if img is None:
                print(f'Failed to read image {imagePath}')
                continue

            # Run inference
            boxes, scores = fd.inference(img)
            # Filter boxes by confidence threshold
            filteredBoxes = []
            filteredScores = []
            for box, score in zip(boxes, scores):
                if score >= confThreshold:
                    filteredBoxes.append(box)
                    filteredScores.append(score)
            numDetections = len(filteredBoxes)

            # Prepare output
            imageID = os.path.splitext(fileName)[0]
            event = os.path.relpath(parent, testDir)  # Preserve directory structure

            # Create output directory if it doesn't exist
            outputDir = os.path.join(outBboxDir, event)
            os.makedirs(outputDir, exist_ok=True)

            outputFile = os.path.join(outputDir, imageID + '.txt')
            with open(outputFile, 'w') as f:
                f.write(f'{imageID}\n')
                f.write(f'{numDetections}\n')
                for box, score in zip(filteredBoxes, filteredScores):
                    xMIN = math.floor(box[0])
                    yMIN = math.floor(box[1])
                    xMAX = math.ceil(box[2])
                    yMAX = math.ceil(box[3])
                    width = xMAX - xMIN
                    height = yMAX - yMIN
                    confidence = min(score, 1.0)
                    # Write in format: xMIN yMIN width height confidence
                    f.write(f'{xMIN} {yMIN} {width} {height} {confidence:.3f}\n')

            counter += 1
            print(f'[{counter}] {fileName} is processed.')

if __name__ == '__main__':
    main()
