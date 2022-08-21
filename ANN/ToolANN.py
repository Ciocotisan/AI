from sklearn import neural_network


class ToolANN:
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
        self.__classifier = None

    def createNetwork(self):
        self.__classifier = neural_network.MLPClassifier(
            hidden_layer_sizes=(self.__noHiddenNeurons,),
            activation=self.__activation,
            max_iter=self.__noEpochs,
            solver="sgd",
            learning_rate_init=0.001,
        )

    def start_learning(self, inputs, outputs):
        self.__classifier.fit(inputs, outputs)

    def make_prediction(self, inputs):
        computedOutputs = self.__classifier.predict(inputs)

        return computedOutputs

    def accuracy_score(self, real, computed):
        trueSum = 0

        for r, c in zip(real, computed):
            if r == c:
                trueSum += 1

        return trueSum / len(real)

