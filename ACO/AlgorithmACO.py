from model.Ant import Ant
from random import random, randint
from threading import Thread
from time import sleep
import threading


class AlgorithmACO:
    def __init__(self, acoParam, problemParam):
        self.__problemParam = problemParam
        self.__acoParam = acoParam
        self.__population = []
        self.__running = True

    def initialize(self):
        self.__population = []

        for _ in range(self.__acoParam["nrAnts"]):
            self.__population.append(Ant(self.__acoParam, self.__problemParam))

    # main algorithm for each ant we make n-1 new steps after I found a road I add last city to form a cycle
    # and after that I update the matrix which store the shared pheromone between ants generations
    def run_aco_alg(self):
        self.__running = True

        th = Thread(target=self.modify_graph, args=())
        th.start()

        for _ in range(self.__problemParam["nrV"] - 1):
            self.move_to_next_city()

        self.add_last_node_in_ants_tour()
        self.__running = False

        self.shared_pheromone_update()

    def modify_graph(self):

        while self.__running == True:

            i = randint(0, self.__problemParam["nrV"] - 1)
            j = randint(0, self.__problemParam["nrV"] - 1)

            while i != j:
                j = randint(0, self.__problemParam["nrV"] - 1)

            self.__problemParam["distance_matrix"][i][j] = randint(1, 100)

            sleep(0.5)

    def get_shared_matrix(self):
        return self.__acoParam["shared_pheromone"]

    # after all ants make a full road we need to modify for each edge the pheromone
    def shared_pheromone_update(self):

        for i in range(self.__problemParam["nrV"]):
            for j in range(self.__problemParam["nrV"]):
                pheromone = 0
                for ant in self.__population:
                    if ant.get_local_pheromone_matrix()[i][j] == 1:
                        pheromone += 1 / ant.get_cost()

                # put new pheromone with degradation
                self.__acoParam["shared_pheromone"][i][j] = (
                    (1 - self.__acoParam["fi"])
                    * self.__acoParam["shared_pheromone"][i][j]
                    + self.__acoParam["fi"] * pheromone
                )

    # we need to move ants to next city
    def move_to_next_city(self):
        for ant in self.__population:
            ant.pick_next_city()

    # get the best ant with lowest cost from population
    def get_best_ant(self):
        best_ant = self.__population[0]

        for current_ant in self.__population:
            if current_ant.get_cost() < best_ant.get_cost():
                best_ant = current_ant

        return best_ant

    # add first node al end position to make a cycle
    def add_last_node_in_ants_tour(self):
        for ant in self.__population:
            ant.add_start_node_at_end()

