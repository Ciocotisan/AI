from Normalization import Normalization
from MyANN import ANN
from ToolANN import ToolANN

inputs = []


myAnn = ANN(
    activation="sigmoid",
    learningRate=0.3,
    noEpochs=300,
    noFeature=4,
    noOutputClass=3,
    noHiddenNeurons=4,
)

from math import exp

print(1 / (1 + exp(-105)))

