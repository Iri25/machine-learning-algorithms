from math import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


class LogisticRegression:

    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # use the gradient descent method
    # simple stochastic GD
    def fit(self, x, y, learningRate=0.001, noEpochs=1000):

        # beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ... + wn * xn
        self.coef_ = [0.0 for _ in range(1 + len(x[0]))]

        # beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ... + wn * xn
        # self.coefficient_ = [random.random() for _ in range(len(x[0]) + 1)]

        for epoch in range(noEpochs):
            # TBA: shuffle the training examples in order to prevent cycles

            cost = 0
            medie = 0.0

            # for each sample from the training data
            for i in range(len(x)):

                # estimate the output
                ycomputed = sigmoid(self.evaluation(x[i], self.coef_))

                # compute the error for the current sample
                crtError = ycomputed - y[i]

                # cost
                cost += pow(crtError, 2)

                # update the coefficients
                for j in range(0, len(x[0])):
                    self.coef_[j + 1] = self.coef_[j + 1] - learningRate * crtError * x[i][j]
                self.coef_[0] = self.coef_[0] - learningRate * crtError * 1

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def evaluation(self, xi, coefficient):
        yi = coefficient[0]
        for j in range(len(xi)):
            yi += coefficient[j + 1] * xi[j]
        return yi

    def predictOneSample(self, sampleFeatures):
        threshold = 0.5
        coefficients = [self.intercept_] + [c for c in self.coef_]
        computedFloatValue = self.evaluation(sampleFeatures, coefficients)
        computed01Value = sigmoid(computedFloatValue)
        computedLabel = 0 if computed01Value < threshold else 1
        return computedLabel

    def predictOneSampleValue(self, sampleFeatures):
        coefficients = [self.intercept_] + [c for c in self.coef_]
        computedFloatValue = self.evaluation(sampleFeatures, coefficients)
        return sigmoid(computedFloatValue)

    def prediction(self, inTest):
        computedLabels = [self.predictOneSample(sample) for sample in inTest]
        return computedLabels
