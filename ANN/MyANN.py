from Neuron import Neuron
from random import random


class ANN:
    def __init__(
        self,
        activation,
        learningRate,
        noEpochs,
        noFeature,
        noOutputClass,
        noHiddenNeurons,
    ):
        self.__activation = activation
        self.__learningRate = learningRate
        self.__noEpochs = noEpochs
        self.__noFeature = noFeature
        self.__noOutputClass = noOutputClass
        self.__noHiddenNeurons = noHiddenNeurons

        self.__network = []

    # date de intrare input si weights de tip liste
    # pe baza inputurilor si a wheiturilor calcurez Inputul
    def compute_weights_for_neuron(self, input, weights):
        result = 0.0
        for i in range(0, len(input)):
            result += input[i] * weights[i]
        # tin cont si de w0
        result += weights[len(input)]
        return result

    def apply_activation_function(self, rez):
        from math import exp

        if self.__activation == "sigmoid":
            return 1.0 / (1.0 + exp(-rez))

    def convertResultInEncodedLabels(self, result):
        sumVal = 0.0

        for el in result:
            sumVal += el

        for i in range(len(result)):
            result[i] /= sumVal

        maxim = max(result)
        for i in range(len(result)):
            if result[i] == maxim:
                result[i] = 1
            else:
                result[i] = 0

    # date de itrare: features = lista de inputuri
    # date de iesire: valorile neuronilor de iesire
    # pornesc din stanga spre dreapta
    def forwardPropagation(self, features):
        for layer in self.__network:
            newFeatures = []
            for neuron in layer:
                sumWeights = self.compute_weights_for_neuron(features, neuron.weights)
                neuron.output = self.apply_activation_function(sumWeights)
                newFeatures.append(neuron.output)
            features = newFeatures

        self.convertResultInEncodedLabels(features)

        return features

    # rezultatul derivarii ce trebuie aplicat petru backwardPropagation
    def get_result_of_derivation(self, val):
        if self.__activation == "sigmoid":
            return val * (1 - val)

    # date de intrare= layerul pe care se aplica functia de softmax
    # aplic functia de softmax pe ultimul neuron ce transforma outputul ptr fiecare neuron din stratul de output
    def applySoftmax(self, layer):
        outSum = 0.0

        for neuron in layer:
            outSum += neuron.output

        for neuron in layer:
            neuron.prob = neuron.output / outSum

    # date de intrare = out : lista de outputuri
    # aici expected trebuie sa fie mapat cu softmax adica in siruri de biti ex:[1,0,0]
    def backwardPropagation(self, out):
        for i in range(len(self.__network) - 1, 0, -1):
            current_layer = self.__network[i]
            errors = []
            if i == len(self.__network) - 1:
                # aplic un softmax pe ultimul layer pentru a calcula probabilitatiile
                self.applySoftmax(current_layer)

                for index in range(len(current_layer)):
                    # calculez eroare ptr fiecare nerorn in parete
                    errors.append(out[index] - current_layer[index].prob)
            else:
                for j in range(0, len(current_layer)):
                    crtError = 0.0
                    nextLayer = self.__network[i + 1]
                    for neuron in nextLayer:
                        crtError += neuron.weights[j] * neuron.delta
                    errors.append(crtError)

            # calculez eroarea care este data inapoi de acest nod
            for j in range(0, len(current_layer)):
                current_layer[j].delta = errors[j] * self.get_result_of_derivation(
                    current_layer[j].output
                )

    # date de intrare: lista de inputuri
    def updateWeights(self, inputsGiven):
        # pentru fiecare layer plecd de la stanga spre dreapta
        for i in range(0, len(self.__network)):
            inputs = inputsGiven[:-1]
            if (
                i > 0
            ):  # daca sunt pe un layer ascuns trebuie preiau outputurile de la layerul anterior
                inputs = [neuron.output for neuron in self.__network[i - 1]]
            for neuron in self.__network[
                i
            ]:  # fac un upadte de weights pentru fiecare neuron din layer
                for j in range(len(inputs)):
                    neuron.weights[j] += self.__learningRate * neuron.delta * inputs[j]
                neuron.weights[-1] += self.__learningRate * neuron.delta

    # realizez o retea neuronala
    def createNetwork(self):

        # pun si neuronul ptr w0
        hiddenLayer = [
            Neuron([random() for _ in range(self.__noFeature + 1)])
            for _ in range(self.__noHiddenNeurons)
        ]
        # adaug in reteaua neoronala layerul ascuns
        self.__network.append(hiddenLayer)

        outputLayer = [
            Neuron([random() for _ in range(self.__noHiddenNeurons + 1)])
            for _ in range(self.__noOutputClass)
        ]
        # adaug si layerul de output
        self.__network.append(outputLayer)

    # functia care incepe antrenarea reletelei neuronale
    def start_learning(self, inputs, outputs):
        for _ in range(0, self.__noEpochs):

            for inp, out in zip(inputs, outputs):
                self.forwardPropagation(inp)
                self.backwardPropagation(out)
                self.updateWeights(inp)

    # functia de determinare a acuratetei
    def accuracy_score(self, real, computed):
        sumCorrect = 0
        for r, c in zip(real, computed):
            indexReal = r.index(max(r))
            indexComputed = c.index(max(c))
            if indexReal == indexComputed:
                sumCorrect += 1

        return sumCorrect / len(real)

    # realizarea unei predictii
    def make_prediction(self, inputs):
        computedOutput = []
        for inp in inputs:
            out = self.forwardPropagation(inp)
            computedOutput.append(out)

        return computedOutput

    # aplic un RMSE pentru calculu erori
    def error_calc(self, realOutputs, computedOutputs):
        from math import sqrt

        partial = 0
        for i in range(len(realOutputs)):
            partial += sum(
                (r - c) ** 2 for r, c in zip(realOutputs[i], computedOutputs[i])
            ) / len(realOutputs[i])

        rmse = sqrt(partial / len(realOutputs))

        return rmse
