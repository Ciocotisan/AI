from model.Ant import Ant
from Utils import Utils
from random import random, seed, randint
from AlgorithmACO import AlgorithmACO
import sys
import time

seed(time.perf_counter)

# utils = Utils("files\\easy_d.txt")
# utils.readFromFileMatrix()

# utils = Utils("files\\medium_d.txt")
# utils.readFromFileMatrix()

# utils = Utils("files\\hard_d.txt")
# utils.readFromFileMatrix()


# utils = Utils("files\\easy_s.txt")
# utils.readFromFileCoordonates()


utils = Utils("files\\medium_s.txt")
utils.readFromFileCoordonates()


# utils = Utils("files\\hard_s.txt")
# utils.readFromFileCoordonates()

n = utils.get_nrV()
matrix = utils.get_matrix()

# print(n)
# print(matrix)

shared_pheromone = [[0 for _ in range(n)] for _ in range(n)]


problemParam = {"nrV": n, "distance_matrix": matrix}

m = randint(n // 2, n)

print(n)
print(m)

acoParam = {
    "nrAnts": m,
    "nrGen": 200,
    "alpha": 0.4,
    "beta": 0.6,
    "q0": 0.7,
    "fi": 0.2,
    "shared_pheromone": shared_pheromone,
}

acoAlg = AlgorithmACO(acoParam, problemParam)

minim_cost = sys.maxsize
road = []


for index in range(acoParam["nrGen"]):

    # for each generation I start with new ants but, pheromone matrix is shared between generations

    acoAlg.initialize()

    acoAlg.run_aco_alg()

    best_ant = acoAlg.get_best_ant()

    if best_ant.get_cost() < minim_cost:
        minim_cost = best_ant.get_cost()
        road = best_ant.get_tour()

    if index % 20 == 0:
        print(
            "In generation "
            + str(index)
            + "\n"
            + "Best ant road= "
            + str(best_ant.get_tour())
            + "\n"
            + "Cost= "
            + str(best_ant.get_cost())
            + "\n"
        )

print("-----------------------------------------------------------------")
print("Cost: " + str(minim_cost))
print("Best road is: " + str(road))


c = 0
for i in range(len(road) - 1):
    c += matrix[road[i]][road[i + 1]]

utils.writeToFile(minim_cost, road)

