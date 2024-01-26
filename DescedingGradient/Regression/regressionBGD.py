import random
import numpy as np
import time


class RegressionBGD:
    def __init__(self):
        self.intercept_ = 0.0
        self.coefficient_ = []

    def fit(self, x, y, learningRate=0.001, noEpochs=1000):
        # beta or w coefficients
        self.coefficient_ = [random.random() for _ in range(len(x[0]) + 1)]
        for epoch in range(noEpochs):
            media = 0.0
            crtError = 0
            cost = 0

            # for each sample from the training data
            for i in range(len(x)):
                random.shuffle(x[i])

                # estimate the output
                ycomputed = self.evaluation(x[i])

                # compute the error for the current sample
                crtError = ycomputed - y[i]

                cost += pow(crtError, 2)

                for j in range(len(x[0])):
                    media += crtError * x[i][j]

            # update the coefficients
            for j in range(0, len(x[0])):
                self.coefficient_[j] = self.coefficient_[j] - learningRate * (media / len(x))
            self.coefficient_[len(x[0])] = self.coefficient_[len(x[0])] - learningRate * crtError * 1
            print((1 / len(x)) * cost)

        self.intercept_ = self.coefficient_[-1]
        self.coefficient_ = self.coefficient_[:-1]

    def evaluation(self, xi):
        yi = self.coefficient_[-1]
        for j in range(len(xi)):
            yi += self.coefficient_[j] * xi[j]
        return yi

    def prediction(self, x):
        yComputed = [self.evaluation(xi) for xi in x]
        return yComputed