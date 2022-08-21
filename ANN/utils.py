import matplotlib.pyplot as plt
import numpy as np
from random import random, randint, shuffle


class Utils:
    def __init__(self):
        self.__trainInputs = None
        self.__trainOutputs = None
        self.__testInputs = None
        self.__testOutputs = None
        self.__inputs = []
        self.__outputs = []
        self.__noData = 0

        self.__feature1 = []
        self.__feature2 = []
        self.__feature3 = []
        self.__feature4 = []
        self.__labelNames = []

    def getFeatures(self):
        return self.__feature1, self.__feature2, self.__feature3, self.__feature4

    def getOutput(self):
        return self.__outputs

    def getInputs(self):
        return self.__inputs

    def getLabelNames(self):
        return self.__labelNames

    def readFile(self, filePath):

        f = open(filePath, "r")

        lines = f.readlines()

        label_output = []

        for line in lines:
            line = line.strip("\n")
            if line != "":
                args = line.split(",")

                self.__feature1.append(float(args[0]))
                self.__feature2.append(float(args[1]))
                self.__feature3.append(float(args[2]))
                self.__feature4.append(float(args[3]))
                label_output.append(args[4])

        lbl_unique = set(label_output)

        for l in lbl_unique:
            self.__labelNames.append(l)

        print(self.__labelNames)

        # mapez outputurile de tip label la cifre
        for el in label_output:

            index = self.__labelNames.index(el)
            bits = []
            for j in range(len(self.__labelNames)):
                if j == index:
                    bits.append(1)
                else:
                    bits.append(0)
            self.__outputs.append(bits)

        for i in range(len(self.__feature1)):
            self.__inputs.append(
                [
                    self.__feature1[i],
                    self.__feature2[i],
                    self.__feature3[i],
                    self.__feature4[i],
                ]
            )

    def loadDigitData(self):
        from sklearn.datasets import load_digits

        data = load_digits()
        inputs = data.images
        outputs = data["target"]
        outputNames = data["target_names"]

        # shuffle the original data
        noData = len(inputs)
        permutation = np.random.permutation(noData)
        self.__inputs = inputs[permutation]
        outputs = outputs[permutation]

        lbl_unique = set(outputs)

        for l in lbl_unique:
            self.__labelNames.append(l)

        for el in outputs:

            index = self.__labelNames.index(el)
            bits = []
            for j in range(len(self.__labelNames)):
                if j == index:
                    bits.append(1)
                else:
                    bits.append(0)
            self.__outputs.append(bits)

    def splitDataInKSubsets(self, k):
        nrOfGroup = len(self.__inputs) // k

        inputs = []
        outputs = []

        indexes = [i for i in range(len(self.__inputs))]
        shuffle(indexes)

        lastIndex = 0

        while lastIndex < len(self.__inputs):
            inp = []
            out = []
            for i in range(nrOfGroup):
                inp.append(self.__inputs[indexes[i + lastIndex]])
                out.append(self.__outputs[indexes[i + lastIndex]])

            inputs.append(inp)
            outputs.append(out)

            lastIndex = i + lastIndex + 1

        return inputs, outputs

    def splitDataInTrainingTest(self):

        np.random.seed(3)
        indexes = [i for i in range(len(self.__inputs))]
        training = np.random.choice(
            indexes, int(0.8 * len(self.__inputs)), replace=False
        )
        test = [i for i in indexes if not i in training]

        self.__trainInputs = [self.__inputs[i] for i in training]
        self.__trainOutputs = [self.__outputs[i] for i in training]

        self.__testInputs = [self.__inputs[i] for i in test]
        self.__testOutputs = [self.__outputs[i] for i in test]

    def plotValues(self, feature1, feature2, feature3, feature4, outputs, title=None):
        for j in range(len(self.__labelNames)):
            w = [feature1[i] for i in range(len(feature1)) if outputs[i] == j]
            x = [feature2[i] for i in range(len(feature1)) if outputs[i] == j]
            y = [feature3[i] for i in range(len(feature1)) if outputs[i] == j]
            z = [feature4[i] for i in range(len(feature1)) if outputs[i] == j]
            plt.scatter(w, x, y, z, label=self.__labelNames[j])

        plt.title(title)
        plt.legend()
        plt.show()

    def histogram_of_data(self, inputs, titleName):
        plt.hist(inputs, 15)
        plt.title(titleName)
        plt.show()

    def plotPredictions(
        self, feature1, feature2, realOutputs, computedOutputs, title, f1Name, f2Name
    ):

        noData = len(feature1)
        for poz in range(len(self.__labelNames)):
            x = [
                feature1[i]
                for i in range(noData)
                if realOutputs[i] == poz and computedOutputs[i] == poz
            ]
            y = [
                feature2[i]
                for i in range(noData)
                if realOutputs[i] == poz and computedOutputs[i] == poz
            ]
            plt.scatter(x, y, label=self.__labelNames[poz] + " (correct)")

        for poz in range(len(self.__labelNames)):
            x = [
                feature1[i]
                for i in range(noData)
                if realOutputs[i] == poz and computedOutputs[i] != poz
            ]
            y = [
                feature2[i]
                for i in range(noData)
                if realOutputs[i] == poz and computedOutputs[i] != poz
            ]
            plt.scatter(x, y, label=self.__labelNames[poz] + " (Incorect)")

        plt.xlabel(f1Name)
        plt.ylabel(f2Name)
        plt.legend()
        plt.title(title)
        plt.show()

    def getTrainingValues(self):
        return self.__trainInputs, self.__trainOutputs

    def getTestValues(self):
        return self.__testInputs, self.__testOutputs

    def plotTrainAndValidation2D(self):
        trainF1input = [x[0] for x in self.__trainInputs]

        testF1input = [x[0] for x in self.__testInputs]

        plt.plot(trainF1input, self.__trainOutputs, "o", label="training")
        plt.plot(testF1input, self.__testOutputs, "g^", label="test")

        plt.title("Train and Test")
        plt.xlabel("GDP capital")
        plt.ylabel("Happiness")
        plt.legend()
        plt.show()

    def plotTrainAndValidation3D(self):

        trainF1input = [x[0] for x in self.__trainInputs]
        trainF2input = [x[1] for x in self.__trainInputs]

        testF1input = [x[0] for x in self.__testInputs]
        testF2input = [x[1] for x in self.__testInputs]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter3D(
            trainF1input,
            trainF2input,
            self.__trainOutputs,
            color="green",
            label="Train data",
        )
        ax.scatter3D(
            testF1input, testF2input, self.__testOutputs, color="red", label="Test data"
        )

        ax.title.set_text("Train data VS Test data")
        ax.legend()
        ax.set_xlabel("Feature1: GDP.per.Capita.")
        ax.set_ylabel("Feature2: Freedom")
        ax.set_zlabel("Output: Happiness.Score")

        plt.show()

    def my_meshgrid(self, x, y):

        x1 = [[val_x for val_x in x] for _ in range(len(y))]
        y1 = [[y[i] for _ in range(len(x))] for i in range(len(y))]

        return x1, y1

    def plotLearntModel_byHand3D(self, w0, w1, w2):
        points = 1000
        trainF1input = [x[0] for x in self.__trainInputs]
        trainF2input = [x[1] for x in self.__trainInputs]

        f1Min = min(trainF1input)

        move = (max(trainF1input) - min(trainF1input)) / points

        input_rndF1 = []

        for _ in range(points):
            input_rndF1.append(f1Min)
            f1Min += move

        f2Min = min(trainF2input)

        move = (max(trainF2input) - min(trainF2input)) / points

        input_rndF2 = []

        for _ in range(points):
            input_rndF2.append(f2Min)
            f2Min += move

        X, Y = self.my_meshgrid(input_rndF1, input_rndF2)

        function = lambda x, y: w0 + x * w1 + y * w2

        F = [
            [function(X[i][j], Y[i][j]) for j in range(len(X[0]))]
            for i in range(len(X))
        ]

        self.plotModel3D(
            np.array(X), np.array(Y), np.array(F), trainF1input, trainF2input
        )

    def plotLearntModel3D(self, w0, w1, w2):

        points = 1000

        trainF1input = [x[0] for x in self.__trainInputs]
        trainF2input = [x[1] for x in self.__trainInputs]

        input_rndF1 = np.linspace(min(trainF1input), max(trainF1input), points)
        input_rndF2 = np.linspace(min(trainF2input), max(trainF2input), points)

        X, Y = np.meshgrid(input_rndF1, input_rndF2)

        function = lambda x, y: w0 + x * w1 + y * w2

        F = function(X, Y)

        self.plotModel3D(X, Y, F, trainF1input, trainF2input)

    def plotModel3D(self, X, Y, F, trainF1input, trainF2input):

        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, F)

        ax.scatter3D(
            trainF1input, trainF2input, self.__trainOutputs, color="green", alpha=1
        )

        ax.title.set_text("Learnt model")
        ax.set_xlabel("Feature1: GDP.per.Capita.")
        ax.set_ylabel("Feature2: Freedom")
        ax.set_zlabel("Output: Happiness.Score")

        plt.show()

    def plotComputedVSReal3D(self, computed):

        testF1input = [x[0] for x in self.__testInputs]
        testF2input = [x[1] for x in self.__testInputs]

        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")

        ax.scatter3D(
            testF1input,
            testF2input,
            self.__testOutputs,
            color="green",
            marker="o",
            label="Real outputs",
        )
        ax.scatter3D(
            testF1input,
            testF2input,
            computed,
            color="red",
            marker="v",
            label="Computed outputs",
        )
        ax.title.set_text("Computed VS Real")
        ax.legend()
        ax.set_xlabel("Feature1: GDP.per.Capita.")
        ax.set_ylabel("Feature2: Freedom")
        ax.set_zlabel("Output: Happiness.Score")

        plt.show()

    def plotLearntModel2D(self, w0, w1):
        points = 1000

        trainF1input = [x[0] for x in self.__trainInputs]
        input_rndF1 = np.linspace(min(trainF1input), max(trainF1input), points)
        function = lambda x: w0 + x * w1

        computed = [function(el) for el in input_rndF1]

        plt.plot(trainF1input, self.__trainOutputs, "ro", label="training data")
        plt.plot(input_rndF1, computed, "b-", label="learnt model")
        plt.title("train data and the learnt model")
        plt.xlabel("GDP capita")
        plt.ylabel("happiness")
        plt.legend()
        plt.show()

    def plotComputedVSReal2D(self, computed):
        testF1input = [x[0] for x in self.__testInputs]
        plt.plot(testF1input, computed, "yo", label="computed test data")
        plt.plot(testF1input, self.__testOutputs, "g^", label="real test data")
        plt.title("Computed test and real test data")
        plt.xlabel("GDP capita")
        plt.ylabel("happiness")
        plt.legend()
        plt.show()

    def plot_raw_vs_normalised_data(
        self, feature1, feature2, feature1Norm, feature2Norm
    ):
        plt.plot(feature1, feature2, "ro", label="raw data")
        plt.plot(feature1Norm, feature2Norm, "b^", label="standardised data")
        plt.title("Data normalization")
        plt.legend()
        plt.show()

    def plot_all_data_in_points(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter3D(self.__feature1, self.__feature2, self.__outputs, color="purple")

        ax.title.set_text("All data")
        ax.set_xlabel("Feature1: GDP.per.Capita.")
        ax.set_ylabel("Feature2: Freedom")
        ax.set_zlabel("Output: Happiness.Score")

        plt.show()

