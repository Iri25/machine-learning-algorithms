import matplotlib.pyplot as plt
import numpy as np


def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def plotLinearRelationship(inputs, outputs):
    plt.plot(inputs, outputs, 'ro')
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.title('GDP capita vs. happiness')
    plt.show()


def plotTrainingTestingData(trainInputs, trainOutputs, testInputs, testOutputs):
    plt.plot(trainInputs, trainOutputs, 'ro', label='training data')  # train data are plotted by red and circle sign
    plt.plot(testInputs, testOutputs, 'g^', label='testing data')  # test data are plotted by green and a triangle sign
    plt.title('train and test data')
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


def plotLearntModel(trainInputs, trainOutputs, w0, w1):
    noOfPoints = 1000
    xref = []
    val = min(trainInputs)
    step = (max(trainInputs) - min(trainInputs)) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val)
        val += step
    yref = [w0 + w1 * el for el in xref]

    plt.plot(trainInputs, trainOutputs, 'ro', label='training data')
    plt.plot(xref, yref, 'b-', label='learnt model')
    plt.title('train data and the learnt model')
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


def plotComputedOutputs(testInputs, testOutputs, computedTestOutputs):
    plt.plot(testInputs, computedTestOutputs, 'yo', label='computed test data')
    plt.plot(testInputs, testOutputs, 'g^', label='real test data')
    plt.title('computed test and real test data')
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


def plotData(x1, y1, x2=None, y2=None, x3=None, y3=None, title=None):
    plt.plot(x1, y1, 'ro', label='train data')
    if x2:
        plt.plot(x2, y2, 'b-', label='learnt model')
    if x3:
        plt.plot(x3, y3, 'g^', label='test data')
    plt.title(title)
    plt.legend()
    plt.show()


def plot3Ddata(x1Train, x2Train, yTrain, x1Model=None, x2Model=None, yModel=None, x1Test=None, x2Test=None, yTest=None,
               title=None):
    ax = plt.axes(projection='3d')
    if x1Train:
         plt.scatter(x1Train, x2Train, yTrain, c='r', marker='o', label='train data')
    if x1Model:
        plt.scatter(x1Model, x2Model, yModel, c='b', marker='_', label='learnt model')
    if x1Test:
         plt.scatter(x1Test, x2Test, yTest, c='g', marker='^', label='test data')
    plt.title(title)
    ax.set_xlabel("capita")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.show()


def plotArea(regressor, trainInput, trainOutput, ax):
    ax.set_xlabel("GDP")
    ax.set_ylabel("Freedom")
    ax.set_zlabel("Happiness")

    x = np.arange(min([trainInput[i][1] for i in range(len(trainInput))]),
                  max([trainInput[i][1] for i in range(len(trainInput))]), 0.1)

    y = np.arange(min([trainInput[i][2] for i in range(len(trainInput))]),
                  max([trainInput[i][2] for i in range(len(trainInput))]), 0.1)
    x, y = np.meshgrid(x, y)

    z = [regressor.prediction2([d1, d2]) for d1, d2 in zip(x, y)]
    z = np.array(z)

    ax.plot_surface(x, y, z.reshape(x.shape), alpha=0.7)
    plt.show()


def plotTrain(regressor, trainInput, trainOutput):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([trainInput[i][1] for i in range(len(trainInput))], [trainInput[i][2] for i in range(len(trainInput))],
               trainOutput, marker='.', color='red')
    plt.title("Train")
    plotArea(regressor, trainInput, trainOutput, ax)


def plotTest(regressor, trainInput, trainOutput, testInput, testOutput):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([testInput[i][0] for i in range(len(testInput))], [testInput[i][1] for i in range(len(testInput))],
               testOutput, marker='^', color='green')
    plt.title("Test")
    plotArea(regressor, trainInput, trainOutput, ax)


def split(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]
    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]
    return trainInputs, trainOutputs, testInputs, testOutputs


def splitdata(inputs, outputs):
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    validationInputs = [inputs[i] for i in validationSample]
    validationOutputs = [outputs[i] for i in validationSample]

    return trainInputs, trainOutputs, validationInputs, validationOutputs
