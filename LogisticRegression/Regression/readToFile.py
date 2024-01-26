from sklearn.datasets import load_iris
import numpy as np


def readDataCode():
    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    outputNames = data['target_names']
    featuresNames = list(data['feature_names'])

    inputs = [[feat[featuresNames.index('sepal length (cm)')], feat[featuresNames.index('petal length (cm)')]] for feat
              in inputs]

    return inputs, outputs


def readDataTool():
    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    outputNames = data['target_names']
    featuresNames = list(data['feature_names'])

    inputs = [[feat[featuresNames.index('sepal length (cm)')], feat[featuresNames.index('sepal width (cm)')],
               feat[featuresNames.index('petal length (cm)')], feat[featuresNames.index('petal width (cm)')]] for feat
              in inputs]

    return inputs, outputs


def splitData(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    validationInputs = [inputs[i] for i in validationSample]
    validationOutputs = [outputs[i] for i in validationSample]

    return trainInputs, trainOutputs, validationInputs, validationOutputs
