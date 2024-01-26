import matplotlib.pyplot as plt


# ------------------------------------------------- Plot Brest Cancer -------------------------------------------------

def plotDataDistributionBreastCancer(inputs, outputs, outputNames, feature1, feature2):
    labels = set(outputs)
    noData = len(inputs)
    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if outputs[i] == crtLabel]
        y = [feature2[i] for i in range(noData) if outputs[i] == crtLabel]
        plt.scatter(x, y, label=outputNames[crtLabel])
    plt.xlabel('mean radius')
    plt.ylabel('mean texture')
    plt.legend()
    plt.show()


def plotDataHistogramBreastCancer(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def plotClassificationDataBrestCancer(feature1, feature2, outputs, outputNames, title=None):
    labels = set(outputs)
    noData = len(feature1)

    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if outputs[i] == crtLabel]
        y = [feature2[i] for i in range(noData) if outputs[i] == crtLabel]
        plt.scatter(x, y, label=outputNames[crtLabel])

    plt.xlabel('mean radius')
    plt.ylabel('mean texture')
    plt.legend()
    plt.title(title)
    plt.show()


def plotPredictionsBrestCancer(feature1, feature2, outputs, realOutputs, computedOutputs, title, labelNames):
    labels = list(set(outputs))
    noData = len(feature1)

    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] == crtLabel]
        y = [feature2[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] == crtLabel]
        plt.scatter(x, y, label=labelNames[crtLabel] + ' (correct)')

    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] != crtLabel]
        y = [feature2[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] != crtLabel]
        plt.scatter(x, y, label=labelNames[crtLabel] + ' (incorrect)')

    plt.xlabel('mean radius')
    plt.ylabel('mean texture')
    plt.legend()
    plt.title(title)
    plt.show()


# ------------------------------------------------- Plot Iris Flowers -------------------------------------------------

def plotTestIrisFlowers1(testInputs, outputTest):
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    for i in range(len(outputTest)):

        x = testInputs[i][0]
        y = testInputs[i][1]
        if outputTest[i] == 0:
            plt.plot(x, y, 'ro')
        if outputTest[i] == 1:
            plt.plot(x, y, "yo")
        if outputTest[i] == 2:
            plt.plot(x, y, "go")

    plt.show()


def plotTestIrisFlowers2(testInputs, outputTest, OutputsOne, OutputsTwo, OutputsThree):
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    for i in range(len(outputTest)):

        x = testInputs[i][0]
        y = testInputs[i][1]

        if OutputsOne[i] == 1:
            plt.plot(x, y, 'r*')
        if OutputsTwo[i] == 1:
            plt.plot(x, y, "y*")
        if OutputsThree[i] == 1:
            plt.plot(x, y, "g*")

    plt.show()
