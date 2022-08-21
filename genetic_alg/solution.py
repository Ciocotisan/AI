from utils import (
    get_adjacent_list,
    generateNewValues,
    fitness,
    devideChromosomeInCommunities,
)
from random import seed, randint
from GA import GA
import time
from CommunityChromosome import Chromosome

adj_list = []
n = get_adjacent_list(adj_list)
n += 1

v = generateNewValues(adj_list)

seed(time.perf_counter)


mask = [randint(0, 1) for _ in range(n)]


# initialise de GA parameters
gaParam = {"popSize": 10, "noGen": 5}
# problem parameters

problParam = {"function": fitness, "noDim": 10, "nrV": n, "adj_list": adj_list}


ga = GA(gaParam, problParam)

ga.initialisation()
ga.evaluation()

max = -99999999999999999
rez = []
for g in range(10):

    # logic alg
    ga.oneGeneration()
    # ga.oneGenerationElitism()
    # ga.oneGenerationSteadyState()

    bestChromo = ga.bestChromosome()
    print(
        "Best solution in generation "
        + str(g)
        + " is: x = "
        + str(bestChromo.repres)
        + " f(x) = "
        + str(bestChromo.fitness)
    )

    if bestChromo.fitness > max:
        max = bestChromo.fitness
        rez = bestChromo.repres

    print()


print(max)
print(rez)

print(devideChromosomeInCommunities(rez))
