from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


class ToolLogisticRegression:
    def __init__(self):
        self.__regresor = None
        self.__intercept = None
        self.__coef = None

    def startLearning(self, inputs, output):
        self.__regresor = LogisticRegression()

        self.__regresor.fit(inputs, output)
        self.__intercept = self.__regresor.intercept_
        self.__coef = self.__regresor.coef_

    def accuracy_score(self, real, computed):
        return accuracy_score(real, computed)

    def error_calc(self, real, computed):
        return mean_squared_error(real, computed)

    def make_prediction(self, inputs):
        return self.__regresor.predict(inputs)

    def getW0(self):
        return self.__intercept

    def getW1(self):
        w1_coef = [x[0] for x in self.__coef]
        return w1_coef

    def getW2(self):
        w2_coef = [x[1] for x in self.__coef]
        return w2_coef

    def getW3(self):
        w2_coef = [x[2] for x in self.__coef]
        return w2_coef

    def getW4(self):
        w2_coef = [x[3] for x in self.__coef]
        return w2_coef
