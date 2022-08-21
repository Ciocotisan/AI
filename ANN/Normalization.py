from sklearn.preprocessing import StandardScaler


class Normalization:
    def normalisation(self, trainData, testData):
        scaler = StandardScaler()
        scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)
        return normalisedTrainData, normalisedTestData
