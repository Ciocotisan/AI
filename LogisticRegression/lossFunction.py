# Mean Square Error(MSE) is greater for learning the outliers in the dataset,
# Mean Absolute Error(MAE) is good to ignore the outliers.
# HuberLos = combination of both MSE and MAE

# Loss(y,f(x)) =  1) 1/2 (y - f(x))^2 , |y-f(x)| <= threshold
#                 2) delta *|y - f(x)| - 1/2 * delta * delta, otherwise

# Huber loss is both MSE and MAE means it is quadratic(MSE) when the error is small else MAE
import math


def huberLoss(real, computed, delta=0.5):

    total_err = 0
    for i in range(len(real)):
        err = abs(real[i] - computed[i])

        if err <= delta:
            huber_error = (err * err) / 2
        else:
            huber_error = delta * err - 0.5 * delta * delta

        total_err += huber_error

    huber = total_err / len(real)

    print(huber)


huberLoss([1, 1, 1, 1, 0, 0, 2, 2], [1, 1, 1, 0, 0, 0, 2, 2])

