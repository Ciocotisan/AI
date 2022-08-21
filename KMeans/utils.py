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

        # print(self.__labelNames)

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

    def readCSVFile(self):
        import csv
        import os

        crtDir = os.getcwd()
        fileName = os.path.join(crtDir, "data", "reviews_mixed.csv")

        data = []
        with open(fileName) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    dataNames = row
                else:
                    data.append(row)
                line_count += 1

        self.__inputs = [data[i][0] for i in range(len(data))]
        self.__outputs = [data[i][1] for i in range(len(data))]
        self.__labelNames = list(set(self.__outputs))

    def getFeaturesWithBag(self, trainFeatures, testFeatures):
        # # representation 1: Bag of Words
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer()

        trainFeatures = vectorizer.fit_transform(trainFeatures)
        testFeatures = vectorizer.transform(testFeatures)

        trainFeatures = trainFeatures.toarray()
        testFeatures = testFeatures.toarray()

        return trainFeatures, testFeatures

    def getFeaturesWithTFIDF(self, trainFeatures, testFeatures):
        # # representation 2: tf-idf features - word granularity
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=50)

        trainFeatures = vectorizer.fit_transform(trainFeatures)
        testFeatures = vectorizer.transform(testFeatures)

        trainFeatures = trainFeatures.toarray()
        testFeatures = testFeatures.toarray()

        return trainFeatures, testFeatures

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

