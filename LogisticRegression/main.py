from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from statistics import mean
from Regression.logisticRegression import LogisticRegression

from Regression.readToFile import splitData, readDataCode, readDataTool

from Regression.utils import plotTestIrisFlowers2, plotTestIrisFlowers1


# ---------------------------------------------------- Brest Cancer ----------------------------------------------------

def normalisationBreastCancer(trainData, testData):
    scalar = StandardScaler()

    if not isinstance(trainData[0], list):
        # encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        # fit only on training data
        scalar.fit(trainData)

        # apply same transformation to train data
        normalisedTrainData = scalar.transform(trainData)

        # apply same transformation to test data
        normalisedTestData = scalar.transform(testData)

        # decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        # fit only on training data
        scalar.fit(trainData)

        # apply same transformation to train data
        normalisedTrainData = scalar.transform(trainData)

        # apply same transformation to test data
        normalisedTestData = scalar.transform(testData)

    return normalisedTrainData, normalisedTestData


def runBreastCancer():
    # print("Brest Cancer")

    data = load_breast_cancer()

    inputs = data['data']
    outputs = data['target']
    outputNames = data['target_names']

    featureNames = list(data['feature_names'])
    feature1 = [feat[featureNames.index('mean radius')] for feat in inputs]
    feature2 = [feat[featureNames.index('mean texture')] for feat in inputs]

    inputs = [[feat[featureNames.index('mean radius')], feat[featureNames.index('mean texture')]] for feat in inputs]

    # plot the data distribution breast cancer
    # plotDataDistributionBreastCancer(inputs, outputs, outputNames, feature1, feature2)

    # plot the data histogram breast cancer
    # plotDataHistogramBreastCancer(feature1, 'mean radius')
    # plotDataHistogramBreastCancer(feature2, 'mean texture')
    # plotDataHistogramBreastCancer(outputs, 'cancer class')

    # plot the classification data breast cancer
    # plotClassificationDataBrestCancer(feature1, feature2, outputs, outputNames, None)

    # split data into train and test subsets
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]

    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    # normalise the features
    trainInputs, testInputs = normalisationBreastCancer(trainInputs, testInputs)

    feature1train = [ex[0] for ex in trainInputs]
    feature2train = [ex[1] for ex in trainInputs]
    feature1test = [ex[0] for ex in testInputs]
    feature2test = [ex[1] for ex in testInputs]

    # plot the normalised data brest cancer
    # plotClassificationDataBrestCancer(feature1train, feature2train, trainOutputs, outputNames,
    # 'normalised train data')

    # using sklearn (variant 1)
    # classifier = linear_model.LogisticRegression()

    # using developed code (variant 2)
    classifier = LogisticRegression()

    # train the classifier (fit in on the training data)
    classifier.fit(trainInputs, trainOutputs)

    # parameters of the linear repressor
    w0, w1, w2 = classifier.intercept_, classifier.coef_[0], classifier.coef_[1]
    # print('\nClassification model brest cancer: y(feat1, feat2) = ', w0, ' + ', w1, ' * feat1 + ', w2, ' * feat2')

    # makes predictions for test data (variant 1)
    # computedTestOutputs = [w0 + w1 * el[0] + w2 * el[1] for el in testInputs]

    # makes predictions for test data (by tool) (variant 2)
    computedTestOutputs = classifier.prediction(testInputs)

    # plot the predictions brest cancer
    # plotPredictionsBrestCancer(feature1test, feature2test, outputs, testOutputs, computedTestOutputs,
    #                            "real test data", outputNames)

    # compute the differences between the predictions and real outputs
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        if t1 != t2:
            error += 1
    error = error / len(testOutputs)
    # print("Classification error (manual) brest cancer: ", error)

    error = 1 - accuracy_score(testOutputs, computedTestOutputs)
    # print("Classification error (tool) brest cancer: ", error)


# runBreastCancer()


# ---------------------------------------------------- Iris Flowers ----------------------------------------------------

# with code
def normalisationIrisFlowers(inputs):
    m = mean(inputs)
    aux = (1 / len(inputs) * sum([(i - m) ** 2 for i in inputs])) ** 0.5
    normalised = [(i - m) / aux for i in inputs]
    return normalised


def errors(inputs_error, outputs_error):
    error = 0
    for i in range(len(inputs_error)):
        error += pow((inputs_error[i] - outputs_error[i]), 2)
    return (1 / len(inputs_error)) * error


def runIrisFlowersCode():
    print("\nIris Flowers with code\n")

    inputs, outputs = readDataCode()

    inputTrain, outputTrain, inputTest, outputTest = splitData(inputs, outputs)

    f1 = normalisationIrisFlowers([x[0] for x in inputTrain])
    f2 = normalisationIrisFlowers([x[1] for x in inputTrain])
    trainInputs = [[f11, f22] for f11, f22 in zip(f1, f2)]

    f1 = normalisationIrisFlowers([x[0] for x in inputTest])
    f2 = normalisationIrisFlowers([x[1] for x in inputTest])
    testInputs = [[f11, f22] for f11, f22 in zip(f1, f2)]

    result1 = LogisticRegression()
    trainOutputsOne = [1 if el == 0 else 0 for el in outputTrain]
    testOutputsOne = [1 if el == 0 else 0 for el in outputTest]
    result1.fit(trainInputs, trainOutputsOne)
    OutputsOne = result1.prediction(testInputs)
    errorOne = errors(OutputsOne, testOutputsOne)

    result2 = LogisticRegression()
    trainOutputsTwo = [1 if el == 1 else 0 for el in outputTrain]
    testOutputsTwo = [1 if el == 1 else 0 for el in outputTest]
    result2.fit(trainInputs, trainOutputsTwo)
    OutputsTwo = result2.prediction(testInputs)
    errorTwo = errors(OutputsTwo, testOutputsTwo)

    result3 = LogisticRegression()
    trainOutputsThree = [1 if el == 2 else 0 for el in outputTrain]
    testOutputsThree = [1 if el == 2 else 0 for el in outputTest]
    result3.fit(trainInputs, trainOutputsThree)
    OutputsThree = result3.prediction(testInputs)
    errorThree = errors(OutputsThree, testOutputsThree)

    print("Error One: " + str(errorOne))
    print("Error Two: " + str(errorTwo))
    print("Error Three: " + str(errorThree))
    print()
    print("Outputs One: " + str(OutputsOne))
    print("Outputs Two: " + str(OutputsTwo))
    print("Outputs Three: " + str(OutputsThree))
    print()
    print("Test: " + str(testOutputsTwo))

    # plot
    plotTestIrisFlowers1(testInputs, outputTest)

    plotTestIrisFlowers2(testInputs, outputTest, OutputsOne, OutputsTwo, OutputsThree)


runIrisFlowersCode()


# with tool
def runIrisFlowersTool():

    print("\nIris Flowers with tool\n")

    inputs, outputs = readDataTool()
    inputTrain, outputTrain, inputTest, outputTest = splitData(inputs, outputs)

    # normalise the dates
    scaler = StandardScaler()
    if not isinstance(inputTrain[0], list):
        inputTrain = [[d] for d in inputTrain]
        inputTest = [[d] for d in inputTest]

        scaler.fit(inputTrain)
        normalisedTrainInput = scaler.transform(inputTrain)
        normalisedTestInput = scaler.transform(inputTest)

        # decode from list
        normalisedTrainInput = [el[0] for el in normalisedTrainInput]
        normalisedTestInput = [el[0] for el in normalisedTestInput]

    else:
        scaler.fit(inputTrain)
        normalisedTrainInput = scaler.transform(inputTrain)
        normalisedTestInput = scaler.transform(inputTest)

    # normalised data: normalisedTrainInput, normalisedTestInput

    logisticRegressionTool = linear_model.LogisticRegression(max_iter=1000)
    logisticRegressionTool.fit(normalisedTrainInput, outputTrain)

    w0, w1, w2, w3, w4 = logisticRegressionTool.intercept_[0], logisticRegressionTool.coef_[0][0], \
                         logisticRegressionTool.coef_[0][1], logisticRegressionTool.coef_[0][2], \
                         logisticRegressionTool.coef_[0][3]

    print('Model SETOSA:  w0 = ', w0, ' w1 = ', w1, ' w2 = ', w2, ' w3 = ', w3, ' w4 = ', w4)

    w0, w1, w2, w3, w4 = logisticRegressionTool.intercept_[1], logisticRegressionTool.coef_[1][0], \
                         logisticRegressionTool.coef_[1][1], logisticRegressionTool.coef_[1][2], \
                         logisticRegressionTool.coef_[1][3]

    print('Model VERSICOLOR:  w0 = ', w0, ' w1 = ', w1, ' w2 = ', w2, ' w3 = ', w3, ' w4 = ', w4)
    w0, w1, w2, w3, w4 = logisticRegressionTool.intercept_[2], logisticRegressionTool.coef_[2][0], \
                         logisticRegressionTool.coef_[2][1], logisticRegressionTool.coef_[2][2], \
                         logisticRegressionTool.coef_[2][3]

    print('Model VIRGINICA:  w0 = ', w0, ' w1 = ', w1, ' w2 = ', w2, ' w3 = ', w3, ' w4 = ', w4)

    print()
    print('Prediction (tool): ', logisticRegressionTool.predict(normalisedTestInput))
    print("Accuracy (tool): ", accuracy_score(outputTest, logisticRegressionTool.predict(normalisedTestInput)))
    error = 1 - accuracy_score(outputTest, logisticRegressionTool.predict(normalisedTestInput))
    print("Classification Error (tool): ", error)


runIrisFlowersTool()
