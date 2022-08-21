from utils import Utils
from Normalization import Normalization
from sklearn.datasets import load_iris
from ToolLogisticRegression import ToolLogisticRegression
from MyLogisticRegression import MyLogisticRegression

# X, y = load_iris(return_X_y=True)

# print(X)
# print(y)
# print(output)


u = Utils()
norm = Normalization()
u.readFile("data\iris.data")

f1, f2, f3, f4 = u.getFeatures()
inputs = u.getInputs()
output = u.getOutput()


# Histogram of features and output
# u.histogram_of_data(f1, "sepal length")
# u.histogram_of_data(f2, "sepal width")
# u.histogram_of_data(f3, "petal length")
# u.histogram_of_data(f4, "petal width")
# u.histogram_of_data(output, "flower_class")


u.splitDataInTrainingTest()

trainInputs, trainOutputs = u.getTrainingValues()
testInputs, testOutputs = u.getTestValues()

f1Train = [x[0] for x in trainInputs]
f2Train = [x[1] for x in trainInputs]
f3Train = [x[2] for x in trainInputs]
f4Train = [x[3] for x in trainInputs]

# u.plotValues(f1Train, f2Train, f3Train, f4Train, trainOutputs, "Train Values")

trainInputs, testInputs = norm.normalisation(trainInputs, testInputs)

# #plot normal data vs normalized
# f1TrainN = [x[0] for x in trainInputs]
# f2TrainN = [x[1] for x in trainInputs]
# f3TrainN = [x[2] for x in trainInputs]
# f4TrainN = [x[3] for x in trainInputs]
# u.histogram_of_data(f1Train, "sepal length")
# u.histogram_of_data(f1TrainN, "sepal lenght norm")
# u.histogram_of_data(f2Train, "petal length")
# u.histogram_of_data(f2TrainN, "petal width norm")

f1Test = [x[0] for x in testInputs]
f2Test = [x[1] for x in testInputs]
f3Test = [x[2] for x in testInputs]
f4Test = [x[3] for x in testInputs]


## MyLogistic

myLogistic = MyLogisticRegression()

myLogistic.start_learning(trainInputs, trainOutputs)

myLogisticOutputComputed = myLogistic.make_prediciton(testInputs)

accMyLogistic = myLogistic.accuracy_score(testOutputs, myLogisticOutputComputed)


errorMyLogistic = myLogistic.error_calc(testOutputs, myLogisticOutputComputed)

u.plotPredictions(
    f1Test,
    f2Test,
    testOutputs,
    myLogisticOutputComputed,
    "Computed vs Real",
    "sepal length",
    "sepal width",
)
u.plotPredictions(
    f3Test,
    f4Test,
    testOutputs,
    myLogisticOutputComputed,
    "Computed vs Real",
    "petal length",
    "petal width",
)


print("Real = ", testOutputs)
print("MyLogistic Computed= ", myLogisticOutputComputed)

print("MyLogistic acuracy score : ", accMyLogistic)
print("MyLogistic error = ", errorMyLogistic)

# w0 = myLogistic.getW0()
# w1 = myLogistic.getW1()
# w2 = myLogistic.getW2()
# w3 = myLogistic.getW3()
# w4 = myLogistic.getW4()

# print(
#     "\nCoeficienti MyLogistic: y(f1,f2,f3,f4) = ",
#     w0,
#     " + ",
#     w1,
#     " * feat1 + ",
#     w2,
#     " * feat2 +",
#     w3,
#     " * feat3 + ",
#     w4,
#     " * feat4",
# )


##TOOL

tool = ToolLogisticRegression()

tool.startLearning(trainInputs, trainOutputs)

toolComputedOutput = tool.make_prediction(testInputs)

accTool = tool.accuracy_score(testOutputs, toolComputedOutput)


errorTool = tool.error_calc(testOutputs, toolComputedOutput)

print("Real = ", testOutputs)
print("Tool Computed= ", toolComputedOutput)

print("Tool acuracy score : ", accTool)
print("Tool error = ", errorTool)


# w0 = tool.getW0()
# w1 = tool.getW1()
# w2 = tool.getW2()
# w3 = tool.getW3()
# w4 = tool.getW4()

# print(
#     "\nCoeficienti Tool: y(f1,f2,f3,f4) = ",
#     w0,
#     " + ",
#     w1,
#     " * feat1 + ",
#     w2,
#     " * feat2 +",
#     w3,
#     " * feat3 + ",
#     w4,
#     " * feat4",
# )
