from math import exp


class MyLogisticRegression:
    def __init__(self):
        self.__coef = []
        self.__intercept = []
        self.__labels = []

    def getCoef(self):
        return self.__coef

    def getIntercept(self):
        return self.__intercept

    def setCoef(self, coef):
        self.__coef = coef

    def setIntercept(self, intercept):
        self.__intercept = intercept

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def computation(self, xi, weights):
        out = weights[-1]

        for j in range(len(xi)):
            out += weights[j] * xi[j]

        return out

    # using Batch GD
    def one_vs_others_batchGD(self, inputs, output, learningRate=0.005, epochs=100):
        weights = [0.0 for _ in range(len(inputs[0]) + 1)]

        for _ in range(epochs):
            error = 0

            for i in range(len(inputs)):
                guess = self.sigmoid(self.computation(inputs[i], weights))
                error += guess - output[i]

            average_error = error / len(inputs)

            for i in range(len(inputs)):
                for j in range(len(inputs[0])):
                    weights[j] = (
                        weights[j] - learningRate * average_error * inputs[i][j]
                    )

            weights[-1] = weights[-1] - learningRate * average_error

        self.__intercept.append(weights[-1])
        self.__coef.append([weights[i] for i in range(len(weights) - 1)])

    # using stochastic GD
    def one_vs_others_stochasticGD(
        self, inputs, output, learningRate=0.005, epochs=1000
    ):
        weights = [0.0 for _ in range(len(inputs[0]) + 1)]

        for _ in range(epochs):

            for i in range(len(inputs)):
                guess = self.sigmoid(self.computation(inputs[i], weights))
                error = guess - output[i]

                for j in range(len(inputs[0])):
                    weights[j] = weights[j] - learningRate * error * inputs[i][j]

                weights[-1] = weights[-1] - learningRate * error

        self.__intercept.append(weights[-1])
        self.__coef.append([weights[i] for i in range(len(weights) - 1)])

    def start_learning(self, inputs, output):
        self.__labels = set(output)

        output1VSAll = []

        for l in self.__labels:
            for out in output:
                if out == l:
                    output1VSAll.append(1)
                else:
                    output1VSAll.append(0)

            self.one_vs_others_stochasticGD(inputs, output1VSAll)
            # self.one_vs_others_batchGD(inputs, output1VSAll)

            output1VSAll = []

    # am 3 ecuatii ale coeficientilor aplic pentru fiecare sample fiecare ecuatie
    # se aplica functia sigmod si se alege valoarea cea mai mare
    # adica daca se alege ecuatia 3 ==> labelul este cel corespunzator pt care s-a calculat aceea ecuatie
    def predictOneSample(self, sample):
        computedValues = []

        for i in range(len(self.__labels)):
            coeficieti = (
                [self.getW1()[i]]
                + [self.getW2()[i]]
                + [self.getW3()[i]]
                + [self.getW4()[i]]
                + [self.getW0()[i]]
            )

            computedValues.append(self.sigmoid(self.computation(sample, coeficieti)))

        maxVal = max(computedValues)
        lblVal = computedValues.index(maxVal)

        return lblVal

    def make_prediciton(self, testInputs):
        computedLabes = [self.predictOneSample(x) for x in testInputs]
        return computedLabes

    def accuracy_score(self, real, computed):
        return sum(
            [1 if real[i] == computed[i] else 0 for i in range(0, len(real))]
        ) / len(real)

    def error_calc(self, real, computed):
        error = 0
        for x, y in zip(computed, real):
            error += (x - y) ** 2

        return error / len(real)

    def getW0(self):
        return self.__intercept

    def getW1(self):
        w1_coef = [x[0] for x in self.__coef]
        return w1_coef

    def getW2(self):
        w2_coef = [x[1] for x in self.__coef]
        return w2_coef

    def getW3(self):
        w2_coef = [x[2] for x in self.__coef]
        return w2_coef

    def getW4(self):
        w2_coef = [x[3] for x in self.__coef]
        return w2_coef

