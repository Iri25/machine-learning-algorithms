from math import sqrt, log, e
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score


def predictionError(realOutputs, computedOutputs):
    errorMAE = sum(abs(r - c) for r, c in zip(realOutputs, computedOutputs)) / len(realOutputs)
    errorRMSE = sqrt(sum((r - c) ** 2 for r, c in zip(realOutputs, computedOutputs)) / len(realOutputs))

    return errorMAE, errorRMSE


# version 1 - using the sklearn functions
def evaluationClassification1(realLabels, computedLabels, labelNames):
    accuracy = accuracy_score(realLabels, computedLabels)
    precision = precision_score(realLabels, computedLabels, average=None, labels=labelNames)
    recall = recall_score(realLabels, computedLabels, average=None, labels=labelNames)

    return accuracy, precision, recall


# version 2 - native code
def evaluationClassification2(realLabels, computedLabels, positive, negative):
    # noCorrect = 0
    # for i in range(0, len(realLabels)):
    #     if (realLabels[i] == computedLabels[i]):
    #         noCorrect += 1
    # accuracy = noCorrect / len(realLabels)
    accuracy = sum([1 if realLabels[i] == computedLabels[i] else 0 for i in range(0, len(realLabels))]) / len(
        realLabels)

    # TP = 0
    # for i in range(0, len(realLabels)):
    #     if (realLabels[i] == 'cat' and computedLabels[i] == 'cat'):
    #         TP += 1
    TP = sum([1 if (realLabels[i] == positive and computedLabels[i] == positive) else 0
              for i in range(len(realLabels))])
    FP = sum([1 if (realLabels[i] == negative and computedLabels[i] == positive) else 0
              for i in range(len(realLabels))])
    TN = sum([1 if (realLabels[i] == negative and computedLabels[i] == negative) else 0
              for i in range(len(realLabels))])
    FN = sum([1 if (realLabels[i] == positive and computedLabels[i] == negative) else 0
              for i in range(len(realLabels))])

    precisionPositive = TP / (TP + FP)
    recallPositive = TP / (TP + FN)
    precisionNegative = TN / (TN + FN)
    recallNegative = TN / (TN + FP)

    return accuracy, precisionPositive, precisionNegative, recallPositive, recallNegative


# version 1 - native code
def transformTheRawOutputsIntoLabels1(realLabels, computedOutputs):
    computedLabels = []
    labelNames = list(set(realLabels))
    for p in computedOutputs:
        probMaxPos = p.index(max(p))
        label = labelNames[probMaxPos]
        computedLabels.append(label)
    return label, computedLabels


# version 2 - by using NumPy library
def transformTheRawOutputsIntoLabels2(realLabels, computedOutputs):
    labelNames = list(set(realLabels))
    computedLabels = [labelNames[np.argmax(p)] for p in computedOutputs]
    return computedLabels


def multiTargetPredictionError(realOutputs, computedOutputs):
    errorMAE = 0
    errorRMSE = 0
    l = len(realOutputs)
    numberOfElements = len(realOutputs[0])
    for i in range(l):
        valueMAE = 0
        valueRMSE = 0
        for j in range(numberOfElements):
            valueMAE = valueMAE + ((abs(realOutputs[i][j] - computedOutputs[i][j])) / numberOfElements)
            valueRMSE = valueRMSE + (((realOutputs[i][j] - computedOutputs[i][j]) ** 2) / numberOfElements)
        errorMAE = errorMAE + (valueMAE / numberOfElements)
        errorRMSE = errorRMSE + (sqrt(valueRMSE) / numberOfElements)
    errorMAE = errorMAE / l
    errorRMSE = errorRMSE / l

    return errorMAE, errorRMSE


def lossPredictionError(realOutputs, computedOutputs):
    lossMAE = []
    lossRMSE = []
    numberOfElements = len(realOutputs[0])
    for i in range(numberOfElements):
        sMAE = 0
        sRMSE = 0
        for r, c in zip(realOutputs, computedOutputs):
            sMAE += abs((r[i] - c[i]))
            sRMSE += (r[i] - c[i]) ** 2
        lossMAE.append(sMAE / len(realOutputs))
        lossRMSE.append(sqrt(sRMSE) / len(realOutputs))

    return lossMAE, lossRMSE


def lossEvaluationClassification(realOutputs, computedOutputs):
    s = 0.0
    numberOfElements = len(realOutputs)
    for i in range(numberOfElements):
        s += realOutputs[i] * log(1e-15 + computedOutputs[i][realOutputs[i]]) + \
             (1 - realOutputs[i]) * log(1e-15 + (1 - computedOutputs[i][realOutputs[i]]))
    loss = 1.0 / numberOfElements * s

    return -loss


def lossMultiLabel(realOutputs, computedOutputs):
    s = 0.0
    numberOfElements = len(realOutputs)
    for i in range(numberOfElements):
        for j in range(len(realOutputs[i])):
            s += realOutputs[i][j] * log(1e-15 + computedOutputs[i][j])
    loss = 1.0 / numberOfElements * s

    return -loss
