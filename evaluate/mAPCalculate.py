"""
This code is used to calculate mAP@50 and mAP@50-95 metrics
"""
import os
import tqdm
import pickle
import argparse
import numpy as np
from bbox import bbox_overlaps
import xml.etree.ElementTree as ET

def parseVocAnnotation(annotationPath):
    tree = ET.parse(annotationPath)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall('object'):
        difficult = obj.find('difficult').text
        # Skip difficult objects if necessary
        # if int(difficult) == 1:
        #     continue
        label = obj.find('name').text
        xml_box = obj.find('bndbox')
        bbox = [float(xml_box.find('xmin').text),
                float(xml_box.find('ymin').text),
                float(xml_box.find('xmax').text),
                float(xml_box.find('ymax').text)]
        boxes.append(bbox)
        labels.append(label)
    return np.array(boxes, dtype=np.float64), labels

def getGTBoxes(gtDir):
    """
    Read ground truth boxes from VOC format annotations.
    Args:
        gtDir: Directory containing VOC XML annotation files.
    Returns:
        gtBoxes: Dictionary mapping image IDs to ground truth boxes.
    """
    gtBoxes = {}
    annotationFiles = [f for f in os.listdir(gtDir) if f.endswith('.xml')]
    for annoFile in tqdm.tqdm(annotationFiles, desc='Reading Ground Truths'):
        annotationPath = os.path.join(gtDir, annoFile)
        imageID = os.path.splitext(annoFile)[0]
        boxes, labels = parseVocAnnotation(annotationPath)
        gtBoxes[imageID] = boxes  # You can also store labels if needed
    return gtBoxes

def readPredFile(filepath):
    imageID = os.path.splitext(os.path.basename(filepath))[0]
    boxes = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Skip the first two lines if they contain image name and number of detections
    idx = 0
    if len(lines) > 0 and lines[0].strip() == imageID:
        idx += 1
    if len(lines) > 1 and lines[1].strip().isdigit():
        idx += 1
    for line in lines[idx:]:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # Read the coordinates as xMIN, yMIN, width, height
        xMIN = float(parts[0])
        yMIN = float(parts[1])
        width = float(parts[2])
        height = float(parts[3])
        score = float(parts[4])
        boxes.append([xMIN, yMIN, width, height, score])
    return imageID, np.array(boxes, dtype=np.float64)

def getPreds(predDir):
    predFiles = [f for f in os.listdir(predDir) if f.endswith('.txt')]
    boxes = dict()
    for predFile in tqdm.tqdm(predFiles, desc='Reading Predictions'):
        filepath = os.path.join(predDir, predFile)
        imageID, preds = readPredFile(filepath)
        # Assuming imageID is extracted correctly
        # Organize predictions by event and imageID if necessary
        # For simplicity, you can ignore events if not applicable
        boxes.setdefault('event', {})[imageID] = preds
    return boxes

def normScore(pred):
    """Normalize scores in predictions."""
    scoreMAX = 0
    scoreMIN = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            scoreMAX = max(_max, scoreMAX)
            scoreMIN = min(_min, scoreMIN)

    diff = scoreMAX - scoreMIN
    if diff == 0:
        diff = 1  # Prevent division by zero
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - scoreMIN) / diff

def imageEval(pred, gt, ignore, iouThresh):
    """Single image evaluation.
    pred: Nx5 (xMIN, yMIN, width, height, score)
    gt: Mx4 (xMIN, yMIN, xMAX, yMAX)
    ignore: Mx1 (ignore flags for gt boxes)
    """
    _pred = pred.copy().astype(np.float64)
    _gt = gt.copy().astype(np.float64)
    predRecall = np.zeros(_pred.shape[0])
    recallList = np.zeros(_gt.shape[0])
    proposalList = np.ones(_pred.shape[0])

    # Convert predictions to xMIN, yMIN, xMAX, yMAX
    _pred[:, 2] = _pred[:, 0] + _pred[:, 2]  # xMAX = xMIN + width
    _pred[:, 3] = _pred[:, 1] + _pred[:, 3]  # yMAX = yMIN + height

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):
        gtOverlap = overlaps[h]
        overlapMAX = gtOverlap.max()
        idxMAX = gtOverlap.argmax()
        if overlapMAX >= iouThresh:
            if ignore[idxMAX] == 0:
                recallList[idxMAX] = -1
                proposalList[h] = -1
            elif recallList[idxMAX] == 0:
                recallList[idxMAX] = 1

        rkeepIdx = np.where(recallList == 1)[0]
        predRecall[h] = len(rkeepIdx)
    return predRecall, proposalList

def imgPRInfo(threshNum, predInfo, proposalList, predRecall):
    prInfo = np.zeros((threshNum, 2)).astype('float')
    for t in range(threshNum):
        thresh = 1 - (t+1)/threshNum
        rIndex = np.where(predInfo[:, 4] >= thresh)[0]
        if len(rIndex) == 0:
            prInfo[t, 0] = 0
            prInfo[t, 1] = 0
        else:
            rIdx = rIndex[-1]
            pIndex = np.where(proposalList[:rIdx+1] == 1)[0]
            prInfo[t, 0] = len(pIndex)
            prInfo[t, 1] = predRecall[rIdx]
    return prInfo

def datasetPRInfo(threshNum, prCurve, faceCounter):
    _prCurve = np.zeros((threshNum, 2))
    for i in range(threshNum):
        if prCurve[i, 0] == 0:
            _prCurve[i, 0] = 0
        else:
            _prCurve[i, 0] = prCurve[i, 1] / prCurve[i, 0]
        _prCurve[i, 1] = prCurve[i, 1] / faceCounter
    return _prCurve

def vocAP(rec, prec):
    """Compute VOC AP given precision and recall."""
    # Append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        if mpre[i - 1] < mpre[i]:
            mpre[i - 1] = mpre[i]

    # Calculate area under PR curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def evaluation(predDir, gtDir):
    pred = getPreds(predDir)
    normScore(pred)
    gtBoxes = getGTBoxes(gtDir)
    iouThresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    apValues = []

    for iouThresh in iouThresholds:
        threshNum = 1000
        faceCounter = 0
        prCurve = np.zeros((threshNum, 2)).astype('float')

        # Use progress bar over images instead of events
        eventNames = list(pred.keys())
        for eventName in eventNames:
            predList = pred[eventName]
            pbar = tqdm.tqdm(predList.keys(), desc='Evaluating at IoU {:.2f}'.format(iouThresh), leave=False)
            for imageName in pbar:
                predInfo = predList[imageName]
                if imageName not in gtBoxes:
                    continue
                gtBboxes = gtBoxes[imageName]
                faceCounter += len(gtBboxes)
                if len(gtBboxes) == 0 or len(predInfo) == 0:
                    continue
                ignore = np.ones(gtBboxes.shape[0])  # All ground truths are valid
                predRecall, proposalList = imageEval(predInfo, gtBboxes, ignore, iouThresh)
                _imgPRInfo = imgPRInfo(threshNum, predInfo, proposalList, predRecall)
                prCurve += _imgPRInfo
            pbar.close()

        prCurve = datasetPRInfo(threshNum, prCurve, faceCounter)
        propose = prCurve[:, 0]
        recall = prCurve[:, 1]
        ap = vocAP(recall, propose)
        aps.append(ap)
        apValues.append("AP@{:.2f}: {:.4f}".format(iouThresh, ap))

    # Print all AP values together
    tqdm.tqdm.write('\n'.join(apValues))
    mAP = np.mean(aps)
    tqdm.tqdm.write("==================== Results ====================")
    tqdm.tqdm.write("mAP@50: {:.4f}".format(aps[0]))
    tqdm.tqdm.write("mAP@50-95: {:.4f}".format(mAP))
    tqdm.tqdm.write("=================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="/home/trdung/Documents/BoschPrj/lightFaceDetectModel/evaluate/evaluationTflite128postQuant")
    parser.add_argument('-g', '--gt', default='/home/trdung/Documents/BoschPrj/00_EDABK_Face_labels/VOCFormat/Annotations')
    args = parser.parse_args()
    evaluation(args.pred, args.gt)
