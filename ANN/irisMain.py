from os import error
from matplotlib.pyplot import subplots_adjust
from utils import Utils
from Normalization import Normalization
from MyANN import ANN
from ToolANN import ToolANN

u = Utils()
norm = Normalization()
u.readFile("data\iris.data")


u.splitDataInTrainingTest()

trainInputs, trainOutputs = u.getTrainingValues()
testInputs, testOutputs = u.getTestValues()

trainInputs, testInputs = norm.normalisation(trainInputs, testInputs)

myAnn = ANN(
    activation="sigmoid",
    learningRate=0.3,
    noEpochs=300,
    noFeature=4,
    noOutputClass=3,
    noHiddenNeurons=4,
)

myAnn.createNetwork()

myAnn.start_learning(trainInputs, trainOutputs)

computed = myAnn.make_prediction(testInputs)

precision = myAnn.accuracy_score(testOutputs, computed)

print("My accuracy= ", precision)

error = myAnn.error_calc(testOutputs, computed)

print("My error= ", error)


toolAnn = ToolANN(
    # logistic ii sigmoid
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

toolAnn.start_learning(trainInputs, toolTrainOut)

toolComputed = toolAnn.make_prediction(testInputs)

print("Tool accuracy:", toolAnn.accuracy_score(toolTestOut, toolComputed))

