from utils import Utils
from Normalization import Normalization
from ToolLogisticRegression import ToolLogisticRegression
from MyLogisticRegression import MyLogisticRegression

u = Utils()
norm = Normalization()

u.readFile("data\iris.data")

k = 5

inputs, outputs = u.splitDataInKSubsets(k)

inputs = norm.normalisationK(inputs)

meanAcc = 0

for i in range(k - 1):

    regr = MyLogisticRegression()
    regr.start_learning(inputs[i], outputs[i])

    computed = regr.make_prediciton(inputs[k - 1])

    acc = regr.accuracy_score(outputs[k - 1], computed)

    meanAcc += acc


meanAcc = meanAcc / (k - 1)

print("MeanAccuracy = ", meanAcc)

