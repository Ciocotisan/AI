from utils import Utils
from Normalization import Normalization
from MyANN import ANN
from ToolANN import ToolANN

u = Utils()
norm = Normalization()
u.loadDigitData()


u.splitDataInTrainingTest()

trainInputs, trainOutputs = u.getTrainingValues()
testInputs, testOutputs = u.getTestValues()


def apply_flatten_on_matrix(matrix):

    list_features = []
    for line in matrix:
        for feat in line:
            list_features.append(feat)
    return list_features


trainInputsFlatten = [apply_flatten_on_matrix(matrix) for matrix in trainInputs]
testInputsFlatten = [apply_flatten_on_matrix(matrix) for matrix in testInputs]


myAnn = ANN(
    activation="sigmoid",
    learningRate=0.5,
    noEpochs=100,
    noFeature=64,
    noOutputClass=10,
    noHiddenNeurons=30,
)


myAnn.createNetwork()

myAnn.start_learning(trainInputsFlatten, trainOutputs)

computed = myAnn.make_prediction(testInputsFlatten)

precision = myAnn.accuracy_score(testOutputs, computed)

print("My accuracy= ", precision)


toolAnn = ToolANN(
    activation="logistic",
    learningRate=0.4,
    noEpochs=1000,
    noFeature=4,
    noOutputClass=3,
    noHiddenNeurons=3,
)

# pentru algoritmul meu outptul este mapat sub forma one-hot encoding in siruri de biti ex:[0,0,1]
# dar outputl folosit de catre neural_network.MLPClassifier va fi de tipul [0,0,1,1,2,0,1,2 ...]

# realizez aceasta conversie de output

toolTrainOut = []

for l in trainOutputs:
    p = l.index(max(l))
    toolTrainOut.append(p)

toolTestOut = []

for l in testOutputs:
    p = l.index(max(l))
    toolTestOut.append(p)

toolAnn.createNetwork()

toolAnn.start_learning(trainInputsFlatten, toolTrainOut)

toolComputed = toolAnn.make_prediction(testInputsFlatten)

print("Tool accuracy:", toolAnn.accuracy_score(toolTestOut, toolComputed))

