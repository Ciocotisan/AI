def loadIrisData():
    from sklearn.datasets import load_iris

    data = load_iris()
    inputs = data["data"]
    outputs = data["target"]
    outputNames = data["target_names"]
    featureNames = list(data["feature_names"])
    feature1 = [feat[featureNames.index("sepal length (cm)")] for feat in inputs]
    feature2 = [feat[featureNames.index("petal length (cm)")] for feat in inputs]
    inputs = [
        [
            feat[featureNames.index("sepal length (cm)")],
            feat[featureNames.index("petal length (cm)")],
        ]
        for feat in inputs
    ]
    return inputs, outputs, outputNames


# step2: split data into train and test
import numpy as np


def splitData(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs


inputs, outputs, outputNames = loadIrisData()
trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)

from sklearn.cluster import KMeans

unsupervisedClassifier = KMeans(n_clusters=3, random_state=0)
unsupervisedClassifier.fit(trainInputs)
computedTestIndexes = unsupervisedClassifier.predict(testInputs)

print(computedTestIndexes)


# computedTestOutputs = [outputNames[value] for value in computedTestIndexes]

# from sklearn.metrics import accuracy_score

# print("acc: ", accuracy_score(testOutputs, computedTestOutputs))

# print(computedTestOutputs)


from MykMeans import kMeans

km = kMeans(3, trainInputs)

km.runAlgo()
computed = km.make_prediction(testInputs)

print(computed)

print("Performance: ", km.dunnIndex())
