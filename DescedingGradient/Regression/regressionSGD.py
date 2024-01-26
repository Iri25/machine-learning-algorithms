class RegressionSGD:

    def __init__(self):
        self.intercept_ = 0.0
        self.coefficient_ = []
        self.coefficient2_ = []

    # simple stochastic GD
    def fit(self, x, y, learningRate=0.001, noEpochs=1000):

        # beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ... + wm *xm
        self.coefficient_ = [0.0 for _ in range(len(x[0]) + 1)]

        # beta or w coefficients
        # self.coefficient_ = [random.random() for _ in range(len(x[0]) + 1)]

        for epoch in range(noEpochs):
            # TBA: shuffle the training examples in order to prevent cycles

            # for each sample from the training data
            for i in range(len(x)):

                # estimate the output
                computedOutput = self.evaluation(x[i])

                # compute the error for the current sample
                crtError = computedOutput - y[i]

                # update the coefficients
                for j in range(0, len(x[0])):
                    self.coefficient_[j] = self.coefficient_[j] - learningRate * crtError * x[i][j]
                self.coefficient_[len(x[0])] = self.coefficient_[len(x[0])] - learningRate * crtError * 1

        self.intercept_ = self.coefficient_[-1]
        self.coefficient_ = self.coefficient_[:-1]

    def evaluation(self, xi):
        yi = self.coefficient_[-1]
        for j in range(len(xi)):
            yi += self.coefficient_[j] * xi[j]
        return yi

    def prediction(self, x):
        computedOutput = [self.evaluation(xi) for xi in x]
        return computedOutput

