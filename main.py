from evaluationML import predictionError, evaluationClassification1, transformTheRawOutputsIntoLabels2, \
    multiTargetPredictionError, lossPredictionError, lossMultiLabel, lossEvaluationClassification

print()

# predictionError
realOutputs = [3, 9.5, 4, 5.1, 6, 7.2, 2, 1]
computedOutputs = [2, 7, 4.5, 6, 3, 8, 3, 1.2]
errorMAE, errorRMSE = predictionError(realOutputs, computedOutputs)
print("----------Prediction Error----------")
print("Error MAE is: " + str(errorMAE))
print("Error RMSE is: " + str(errorRMSE))
print()

# evaluationClassification1 with labels
realLabelsEquable = ['cat', 'cat', 'dog', 'dog', 'cat', 'dog', 'cat', 'cat', 'dog', 'dog', 'cat', 'dog']
computedLabelsEquable = ['cat', 'dog', 'dog', 'cat', 'cat', 'dog', 'cat', 'dog', 'dog', 'cat', 'cat', 'dog']
labelNamesEquable = ['cat', 'dog']
accuracy, precision, recall = evaluationClassification1(realLabelsEquable, computedLabelsEquable, labelNamesEquable)
print("----------Evaluation Classification 1 With Label Type Outputs Equable----------")
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))
print("Recall is: " + str(recall))
print()

realLabelsUnequal = ['sick', 'sick', 'sick', 'sick', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy',
                     'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy']
computedLabelsUnequal = ['sick', 'sick', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy',
                         'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy',
                         'sick']
labelNamesUnequal = ['sick', 'healthy']
accuracy, precision, recall = evaluationClassification1(realLabelsUnequal, computedLabelsUnequal, labelNamesUnequal)
print("----------Evaluation Classification 1 With Label Type Outputs Unequal----------")
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))
print("Recall is: " + str(recall))
print()

# evaluationClassification2 with labels
# realLabels = ['cat', 'cat', 'dog', 'dog', 'cat', 'dog']
# computedLabels = ['cat', 'dog', 'dog', 'cat', 'cat', 'dog']
# positive = ['cat']
# negative = ['dog']
# accuracy, precisionPositive, precisionNegative, recallPositive, recallNegative =
# evaluationClassification2(realLabels, computedLabels, positive, negative)
# print("----------Evaluation Classification 2----------")
# print("Accuracy is: " + str(accuracy))
# print("Precision Positive is: " + str(precisionPositive))
# print("Precision Negative is: " + str(precisionNegative))
# print("Recall Positive is: " + str(recallPositive))
# print("Recall Negative is: " + str(recallNegative))
# print()

# evaluationClassification1 with probability
realLabels = ['sick', 'sick', 'sick', 'sick', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy',
              'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy']
computedOutputs = [[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1], [0.7, 0.3], [0.4, 0.6], [0.4, 0.6], [0.5, 0.5],
                   [0.2, 0.8], [0.8, 0.2], [0.9, 0.1], [0.7, 0.3], [0.9, 0.1], [0.9, 0.1], [0.4, 0.6], [0.9, 0.1],
                   [0.7, 0.3]]
labelNames = ['sick', 'healthy']
computedLabels = transformTheRawOutputsIntoLabels2(realLabels, computedOutputs)
accuracy, precision, recall = evaluationClassification1(realLabels, computedLabels, labelNames)
print("----------Evaluation Classification 1 With Probability Type Outputs----------")
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))
print("Recall is: " + str(recall))
print()

# multiTargetPredictionError
realOutputs = [[1, 4, 5], [5, 8, 10], [2, 8, 12], [7, 9, 3], [6, 9, 1]]
computedOutputs = [[1, 3.9, 2.2], [4.2, 5, 2], [8, 5, 1], [6.3, 4.7, 2], [8, 5, 3.3]]
errorMAE, errorRMSE = multiTargetPredictionError(realOutputs, computedOutputs)
print("----------Multi - Target Prediction Error----------")
print("Error MAE is: " + str(errorMAE))
print("Error RMSE is: " + str(errorRMSE))
print()

# multiClassEvaluationClassification
realLabels = ['guitar', 'guitar', 'guitar', 'drums', 'drums', 'violin', 'drums', 'drums', 'violin', 'violin']
computedLabels = ['guitar', 'guitar', 'violin', 'drums', 'drums', 'violin', 'drums', 'violin', 'violin', 'guitar']
labelNames = ['guitar', 'drums', 'violin']
accuracy, precision, recall = evaluationClassification1(realLabels, computedLabels, labelNames)
print("----------Multi - Class Evaluation Classification ----------")
print("Accuracy is: " + str(accuracy))
print("Precision is: " + str(precision))
print("Recall is: " + str(recall))
print()

# lossPredictionError
realOutputs = [[1, 4, 5], [5, 8, 10], [2, 8, 12], [7, 9, 3], [6, 9, 1]]
computedOutputs = [[1, 3.9, 2.2], [4.2, 5, 2], [8, 5, 1], [6.3, 4.7, 2], [8, 5, 3.3]]
errorMAE, errorRMSE = lossPredictionError(realOutputs, computedOutputs)
print("----------Loss Prediction Error----------")
print("Loss MAE is: " + str(errorMAE))
print("Loss RMSE is: " + str(errorRMSE))
print()

# lossEvaluationClassification
realOutputs = [1, 0, 1, 0]
computedOutputs = [[0.2, 0.8], [0.8, 0.2], [0.9, 0.1], [0.7, 0.3]]
lossClassification = lossEvaluationClassification(realOutputs, computedOutputs)
print("----------Loss Evaluation Classification----------")
print("Loss is: " + str(lossClassification))
print()

# lossMultiLabel
realOutputs = [[0, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]]
computedOutputs = [[0, 1, 0, 0], [0, 1, 0, 1], [1, 0, 0, 1], [1, 0, 1, 1]]
lossLabel = lossMultiLabel(realOutputs, computedOutputs)
print("----------Loss Multi Label----------")
print("Loss is: " + str(lossLabel))
print()