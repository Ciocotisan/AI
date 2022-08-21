from os import error
from utils import Utils
from Normalization import Normalization


u = Utils()
norm = Normalization()
u.readFile("data\iris.data")


u.splitDataInTrainingTest()

trainInputs, trainOutputs = u.getTrainingValues()
testInputs, testOutputs = u.getTestValues()

trainInputs, testInputs = norm.normalisation(trainInputs, testInputs)

from MykMeans import kMeans

km = kMeans(3, trainInputs)

km.runAlgo()

computed = km.make_prediction(testInputs)

print("Performance: ", km.dunnIndex())

print("Computed :", computed)
