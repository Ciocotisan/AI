# from fcOptimisGA.RealChromosome import Chromosome
from RealChromosome import Chromosome
from random import seed
from GA import GA
import random

# define the function
import math

MIN = -5
MAX = 5


def fcEval(x):
    # sphere function
    # val = sum(xi ** 2 for xi in x)

    # val = term1 - term2 + 1

    return random.randint(10, 1000)


seed(1)


# initialise de GA parameters
gaParam = {"popSize": 10, "noGen": 3, "pc": 0.8, "pm": 0.1}
# problem parameters
problParam = {"min": MIN, "max": MAX, "function": fcEval, "noDim": 10, "noBits": 8}

# store the best/average solution of each iteration (for a final plot used to anlyse the GA's convergence)
allBestFitnesses = []
allAvgFitnesses = []
generations = []


ga = GA(gaParam, problParam)
ga.initialisation()
ga.evaluation()

for g in range(gaParam["noGen"]):

    # logic alg
    # ga.oneGeneration()
    # ga.oneGenerationElitism()
    ga.oneGenerationSteadyState()

    bestChromo = ga.bestChromosome()
    print(
        "Best solution in generation "
        + str(g)
        + " is: x = "
        + str(bestChromo.repres)
        + " f(x) = "
        + str(bestChromo.fitness)
    )

    print()
