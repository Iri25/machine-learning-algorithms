import os
from cmath import sqrt

import matplotlib.pyplot as plt

from Regression.readToFile import loadData, loadData2
from Regression.regression import Regression

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

from Regression.utils import plotData


def run():
    currentDirectory = os.getcwd()
    filePath = os.path.join(currentDirectory, 'Data', 'world_happiness_report_2017.csv')

    inputs, outputs = loadData(filePath, 'Economy..GDP.per.Capita.', 'Happiness.Score')
    input, output = loadData2('Data/world_happiness_report_2017.csv',
                              ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')

    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    validationInputs = [inputs[i] for i in validationSample]
    validationOutputs = [outputs[i] for i in validationSample]

    indexes2 = [i for i in range(len(input))]
    trainIndex = np.random.choice(indexes2, int(0.8 * len(input)))
    testIndex = [i for i in indexes2 if i not in trainIndex]

    trainInput = [([1] + input[i]) for i in trainIndex]
    trainOutput = [[output[i]] for i in trainIndex]

    testInput = [input[i] for i in testIndex]
    testOutput = [output[i] for i in testIndex]

    # ------------------------------------------------------------------------------------------------------------------
    # training data preparation (the sklearn linear model requires as input training data as noSamples x noFeatures
    # array; in the current case, the input must be a matrix of len(trainInputs) lines and one columns
    # (a single feature is used in this problem))
    # xx = [[el] for el in trainInputs]

    # model initialisation 1
    # regressor = linear_model.LinearRegression()
    # training the model by using the training inputs and known training outputs
    # regressor.fit(xx, trainOutputs)
    # save the model parameters
    # w0, w1 = regressor.intercept_, regressor.coefficient_[0]
    # print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x')
    # ------------------------------------------------------------------------------------------------------------------
    # model initialisation 2
    # regressor = Regression()
    # training the model by using the training inputs and known training outputs
    # regressor.fit(trainInputs, trainOutputs)
    # save the model parameters
    # w0, w1 = regressor.intercept_, regressor.coefficient_
    # print('\nThe learnt model: f(x) = ', w0, ' + ', w1, ' * x\n')

    # makes predictions for test data (manual)
    # testInputs = [regressor.intercept_ + trainInputs[i] for i in range(len(trainInputs))]
    # computedTestOutputs = [w0 + w1 * el for el in trainInputs]

    # makes predictions for test data (by tool)
    # computedValidationOutputs = regressor.predict([[x] for x in validationInputs])

    # "manual" computation
    # error = 0.0
    # for t1, t2 in zip(computedTestOutputs, validationOutputs):
    #     error += (t1 - t2) ** 2
    # error = error / len(validationOutputs)
    # w1 = error
    # print('Prediction error (manual): ', error)
    #
    # # "tool" computation
    # error = mean_squared_error(validationOutputs, computedValidationOutputs)
    # w2 = error
    # print('Prediction error (tool):  ', error)
    # ------------------------------------------------------------------------------------------------------------------
    # model initialisation 3

    # ------------------------
    # TOOL regression
    # ------------------------
    regressor = LinearRegression()
    regressor.fit(trainInput, trainOutput)

    testInputs = [[regressor.intercept_] + testInput[i] for i in range(len(testInput))]
    computedDataOutput = regressor.predict(testInputs)

    error = mean_squared_error(testOutput, computedDataOutput)
    w1 = error
    print("Tool error: ", error)

    # ----------------------------
    # MANUAL regression
    # ----------------------------
    regressor = Regression()
    regressor.fit2(trainInput, trainOutput)

    computedOutput = [regressor.prediction2(data) for data in testInput]

    error = 0
    for t1, t2 in zip(computedOutput, testOutput):
        error += (t1 - t2) ** 2
    error = error / len(testOutput)
    w2 = error
    print("Manual error: ", error)

    # plot 2D
    plotData(inputs, outputs, [], [], [], [], 'capita vs. hapiness')

    plotData(trainInputs, trainOutputs, [], [], validationInputs, validationOutputs, "train and test data")

    noOfPoints = 1000
    xref = []
    val = min(trainInputs)
    step = (max(trainInputs) - min(trainInputs)) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val)
        val += step
    w0 = 2
    yref = [w0 + w1 * el for el in xref]
    plotData(trainInputs, trainOutputs, xref, yref, [], [], title="train data and model")

    # plotData([], [], validationInputs, computedDataOutput, validationInputs, validationOutputs,
    #        "predictions vs real test data")

    # plot 3D
    trainInputs1 = trainInputs
    trainInputs2 = trainOutputs

    xref = np.linspace(min(trainInputs1), max(trainInputs1), 1000)
    yref = np.linspace(min(trainInputs2), max(trainInputs2), 1000)
    zref = []

    x_surf, y_surf = np.meshgrid(xref, yref)

    for el2 in range(len(yref)):
        for el in range(len(xref)):
            zref.append([w0 + w1 * xref[el] + w2 * yref[el2]])
    z_vals = np.array(zref)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(trainInputs1, trainInputs2, trainOutputs, label='train data')
    ax.plot_surface(x_surf, y_surf, z_vals.reshape(x_surf.shape), color='None', alpha=0.5)
    plt.legend()
    plt.xlabel('gdp capita')
    plt.ylabel('freedom')
    plt.title('train data and model')
    plt.show()


run()
