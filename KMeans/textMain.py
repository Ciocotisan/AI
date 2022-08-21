from os import error
from utils import Utils
from Normalization import Normalization
from MyANN import ANN
from MyLogisticRegression import MyLogisticRegression


u = Utils()
norm = Normalization()

u.readCSVFile()

labelNames = u.getLabelNames()

u.splitDataInTrainingTest()

trainInputs, trainOutputs = u.getTrainingValues()
testInputs, testOutputs = u.getTestValues()

# trainFeatures, testFeatures = u.getFeaturesWithBag(trainInputs, testInputs)
trainFeatures, testFeatures = u.getFeaturesWithTFIDF(trainInputs, testInputs)

# from sklearn.cluster import KMeans
# from sklearn.metrics import davies_bouldin_score

# toolkMeans = KMeans(n_clusters=2, random_state=0)
# toolkMeans.fit(trainFeatures)
# computedTool = toolkMeans.predict(testFeatures)

from MykMeans import kMeans

# two clusters 1)positive 2)negative
km = kMeans(2, trainFeatures)

km.runAlgo()

computed = km.make_prediction(testFeatures)

print("Performance: ", km.dunnIndex())

print("Computed :", computed)

trainOutputsMapped = []
for el in trainOutputs:
    trainOutputsMapped.append(labelNames.index(el))

testOutputsMapped = []
for el in testOutputs:
    testOutputsMapped.append(labelNames.index(el))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

regresor = LogisticRegression()
regresor.fit(trainFeatures, trainOutputsMapped)

computed = regresor.predict(testFeatures)

print(accuracy_score(testOutputsMapped, computed))

# myLogistic = MyLogisticRegression()

# myLogistic.start_learning(trainFeatures, trainOutputsMapped)

# myLogisticOutputComputed = myLogistic.make_prediciton(trainInputs)

# accMyLogistic = myLogistic.accuracy_score(testOutputsMapped, myLogisticOutputComputed)
