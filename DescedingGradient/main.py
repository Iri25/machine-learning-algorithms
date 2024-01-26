import os
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from Regression.readToFile import readData, readDataMoreInputs
from Regression.regressionBGD import RegressionBGD
from Regression.regressionSGD import RegressionSGD
from Regression.utils import plotDataHistogram, plotLinearRelationship, plotTrainingTestingData, plotLearntModel, \
    plotComputedOutputs, plotData, plotTrain, plotTest, split, splitdata, plot3Ddata


# -------------------------------------------------------- SGD --------------------------------------------------------
# SGD UNVARIED

def normalisationSGD_Unvaried(inputs):
    min1 = min(inputs)
    max1 = max(inputs)
    return [(y - min1) / (max1 - min1) for y in inputs]


def runSGD_Unvaried():
    currentDirectory = os.getcwd()
    filePath = os.path.join(currentDirectory, 'Data', 'world_happiness_report_2017.csv')

    inputs, outputs = readData(filePath, 'Economy..GDP.per.Capita.', 'Happiness.Score')

    # plotData(inputs, outputs, [], [], [], [], 'capita vs. hapiness')

    # split data into training data (80%) and testing data (20%)
    trainInputs, trainOutputs, testInputs, testOutputs = split(inputs, outputs)

    trainInputs = normalisationSGD_Unvaried(trainInputs)
    testInputs = normalisationSGD_Unvaried(testInputs)

    # plotData(trainInputs, trainOutputs, [], [], testInputs, testOutputs, "train and test data")

    # training step
    xx = [[el] for el in trainInputs]

    # regressor = linear_model.RegressionSGD(max_iter =  10000)
    regressor = RegressionSGD()
    regressor.fit(xx, trainOutputs)
    w0, w1 = regressor.intercept_, regressor.coefficient_[0]

    # plot the model
    noOfPoints = 1000
    xref = []
    val = min(trainInputs)
    step = (max(trainInputs) - min(trainInputs)) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val)
        val += step
    yref = [w0 + w1 * el for el in xref]
    # plotData(trainInputs, trainOutputs, xref, yref, [], [], title="train data and model")

    # makes predictions for test data
    # computedTestOutputs = [w0 + w1 * el for el in testInputs]

    # makes predictions for test data (by tool)
    computedTestOutputs = regressor.prediction([[x] for x in testInputs])

    # plotData([], [], testInputs, computedTestOutputs, testInputs, testOutputs, "predictions vs real test data")

    # compute the differences between the predictions and real outputs
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print("\nPrediction SGD unvaried error (manual): ", error)

    error = mean_squared_error(testOutputs, computedTestOutputs)
    print("Prediction SGD unvaried error (tool): ", error)


runSGD_Unvaried()


# SGD BIVARIANT
def normalisationSGD_Bivariant(trainData, testData):
    standardScaler = StandardScaler()
    if not isinstance(trainData[0], list):
        # encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        # fit only on training data
        standardScaler.fit(trainData)

        # apply same transformation to train data
        normalisedTrainData = standardScaler.transform(trainData)

        # apply same transformation to test data
        normalisedTestData = standardScaler.transform(testData)

        # decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        # fit only on training data
        standardScaler.fit(trainData)

        # apply same transformation to train data
        normalisedTrainData = standardScaler.transform(trainData)

        # apply same transformation to test data
        normalisedTestData = standardScaler.transform(testData)

    return normalisedTrainData, normalisedTestData


def runSGD_Bivariant():
    currentDirectory = os.getcwd()
    filePath = os.path.join(currentDirectory, 'Data', 'world_happiness_report_2017.csv')

    inputs, outputs = readDataMoreInputs(filePath, ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')

    feature1 = [ex[0] for ex in inputs]
    feature2 = [ex[1] for ex in inputs]

    # check the liniarity (to check that a linear relationship exists between the dependent variable (y = happiness)
    # and the independent variables (x1 = capita, x2 = freedom).)
    # plot3Ddata(feature1, feature2, outputs, [], [], [], [], [], [], 'capita vs freedom vs happiness')

    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    trainInputs, testInputs = normalisationSGD_Bivariant(trainInputs, testInputs)
    trainOutputs, testOutputs = normalisationSGD_Bivariant(trainOutputs, testOutputs)

    feature1train = [ex[0] for ex in trainInputs]
    feature2train = [ex[1] for ex in trainInputs]

    feature1test = [ex[0] for ex in testInputs]
    feature2test = [ex[1] for ex in testInputs]

    # plot3Ddata(feature1train, feature2train, trainOutputs, [], [], [], feature1test, feature2test, testOutputs,
    #              "train and test data after normalisation")

    # model initialisation
    regressor = RegressionSGD()

    regressor.fit(trainInputs, trainOutputs)
    # print(regressor.coefficient_)
    # print(regressor.intercept_)

    # parameters of the liniar regressor
    w0, w1, w2 = regressor.intercept_, regressor.coefficient_[0], regressor.coefficient_[1]

    # numerical representation of the regressor model
    noOfPoints = 50
    xref1 = []
    val = min(feature1)
    step1 = (max(feature1) - min(feature1)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1

    xref2 = []
    val = min(feature2)
    step2 = (max(feature2) - min(feature2)) / noOfPoints
    for _ in range(1, noOfPoints):
        aux = val
        for _ in range(1, noOfPoints):
            xref2.append(aux)
            aux += step2
    yref = [w0 + w1 * el1 + w2 * el2 for el1, el2 in zip(xref1, xref2)]
    # plot3Ddata(feature1train, feature2train, trainOutputs, xref1, xref2, yref, [], [], [],
    #            'train data and the learnt model')

    # makes predictions for test data (by tool)
    computedTestOutputs = regressor.prediction([[x] for x in testInputs])

    # compute the differences between the predictions and real outputs
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print("\nPrediction SGD bivariant error (manual):", error)


runSGD_Bivariant()


# -------------------------------------------------------- BGD --------------------------------------------------------

# BGD UNVARIED
def normalisationBGD_Unvaried(inputs):
    min1 = min(inputs)
    max1 = max(inputs)
    return [(y - min1) / (max1 - min1) for y in inputs]


def runBGD_Unvaried():
    currentDirectory = os.getcwd()
    filePath = os.path.join(currentDirectory, 'Data', 'world_happiness_report_2017.csv')

    inputs, outputs = readData(filePath, 'Economy..GDP.per.Capita.', 'Happiness.Score')

    # plotData(inputs, outputs, [], [], [], [], 'capita vs. hapiness')

    trainInputs, trainOutputs, validationInputs, validationOutputs = splitdata(inputs, outputs)

    # normalizer
    f1 = normalisationBGD_Unvaried(trainInputs)

    validatingInputs = normalisationBGD_Unvaried(validationInputs)

    # plotData(trainInputs, trainOutputs, [], [], validationInputs, validationOutputs, "train and test data")

    # training step
    xx = [[el] for el in f1]

    print("\n")
    #regressor = linear_model.RegressionBGD(max_iter=1000)
    regressor = RegressionBGD()
    regressor.fit(xx, trainOutputs)

    w0, w1 = regressor.intercept_, regressor.coefficient_[0]

    # plot the model
    noOfPoints = 100000
    xref = []
    val = min(trainInputs)
    step = (max(trainInputs) - min(trainInputs)) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val)
        val += step
    yref = [w0 + w1 * el for el in xref]
    # plotData(trainInputs, trainOutputs, xref, yref, [], [], title="train data and model")

    computedTestOutputs = regressor.prediction([[x] for x in validationInputs])
    # plotData([], [], validationInputs, computedTestOutputs, validationInputs, validationOutputs,
    #       "predictions vs real test data")

    error = 0.0
    for t1, t2 in zip(computedTestOutputs, validationOutputs):
        error += (t1 - t2) ** 2
    error = error / len(validationOutputs)
    print("\nPrediction BGD unvaried error (manual): ", error)

    computedValidationOutputs = [w0 + w1 * element for element in validatingInputs]

    error = mean_squared_error(validationOutputs, computedTestOutputs)
    print("Prediction BGD unvaried error (tool): ", error)


runBGD_Unvaried()
