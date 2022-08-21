from random import randint
from utils import generateNewValues


class Chromosome:
    def __init__(self, problParam=None):
        self.__problParam = problParam

        self.__repres = generateNewValues(problParam["adj_list"])

        self.__fitness = 0.0

    @property
    def repres(self):
        return self.__repres

    @property
    def fitness(self):
        return self.__fitness

    @repres.setter
    def repres(self, l=[]):
        self.__repres = l

    @fitness.setter
    def fitness(self, fit=0.0):
        self.__fitness = fit

    # uniform cross over
    # consider: chromosome self este parintele1 iar chromosome c este parintele2
    # se cosidera o masca de biti de lugimea nr de noduri
    # daca mask[i] = 1 ==> child[i] = p1[i] altfel child[i] = p2[i]
    def crossover(self, c):

        child = []
        mask = [randint(0, 1) for _ in range(len(self.__repres))]

        for i in range(len(self.__repres)):
            if mask[i] == 1:
                child.append(self.__repres[i])
            else:
                child.append(c.__repres[i])

        offspring = Chromosome(c.__problParam)
        offspring.repres = child

        return offspring

    # la mutatie aleg random o pozite din vect chromosome si schimb aceea pozitie cu un alt vecin
    def mutation(self):

        pos = randint(0, len(self.__repres) - 1)

        # daca am ales un nod care in lista de adiacenta are un singur vecin aleg altul
        # deoarece prin mutatie eu trebuie sa schimb unui anumit nod i vecinul curent
        while len(self.__problParam["adj_list"][pos]) == 1:
            pos = randint(0, len(self.__repres) - 1)

        currentNeighbor = self.__repres[pos]

        posNewNeighbor = randint(0, len(self.__problParam["adj_list"][pos]) - 1)

        # verific ca nu cumva sa inlocuiesc vecinul lui i cu exact acelasi vecin daca are cel putin 2
        # vreau sa il pun celalalt
        while self.__problParam["adj_list"][pos][posNewNeighbor] == currentNeighbor:
            posNewNeighbor = randint(0, len(self.__problParam["adj_list"][pos]) - 1)

        self.__repres[pos] = self.__problParam["adj_list"][pos][posNewNeighbor]

    def __str__(self):
        return "\nChromo: " + str(self.__repres) + " has fit: " + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness
