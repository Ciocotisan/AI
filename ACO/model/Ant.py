from random import randint, random, choices


class Ant:
    def __init__(self, acoParam, problemParam):
        self.__acoParam = acoParam
        self.__problemParam = problemParam

        self.__tour = [randint(0, self.__problemParam["nrV"] - 1)]
        self.__cost = 0

        self.__pheromone_matrix = [
            [0 for _ in range(self.__problemParam["nrV"])]
            for _ in range(self.__problemParam["nrV"])
        ]

    def get_tour(self):
        return self.__tour

    def get_cost(self):
        return self.__cost

    def get_local_pheromone_matrix(self):
        return self.__pheromone_matrix

    # add first node at the end to create a cycle
    def add_start_node_at_end(self):
        self.__tour.append(self.__tour[0])

        self.__cost += self.__problemParam["distance_matrix"][self.__tour[-2]][
            self.__tour[-1]
        ]

        self.__pheromone_matrix[self.__tour[-2]][self.__tour[-1]] = 1
        self.__pheromone_matrix[self.__tour[-1]][self.__tour[-2]] = 1

    def pick_next_city(self):
        # if I am in city i for each city j which is not visited I calculate value tau(i,j)^alpha * niu(i,j)^beta and store it into a dictionary
        # tau(i,j) pheromone
        # niu(i,j) vizibility

        store = {}

        amount = 0

        for city in range(self.__problemParam["nrV"]):
            if city not in self.__tour:
                # tau could be 0 if on that road didn't go some ant before and when I need to multiply tau and niu I need to make
                # the difference bettween short row and long row
                tau = (
                    self.__acoParam["shared_pheromone"][self.__tour[-1]][city]
                    ** self.__acoParam["alpha"]
                )

                if tau == 0:
                    tau = 1.0

                niu = (
                    self.__problemParam["distance_matrix"][self.__tour[-1]][city]
                    ** self.__acoParam["beta"]
                )

                value = tau * niu

                if value == 0:
                    print("ESTE VALUE 0 ")
                store[city] = value
                amount += value

        q = random()
        next_city = -1

        if q <= self.__acoParam["q0"]:

            maxim = -1

            for possible_city in store:
                if store[possible_city] > maxim:
                    maxim = store[possible_city]
                    next_city = possible_city

        else:

            possible_city_list = []
            weights = []

            for possible_city in store:
                possible_city_list.append(possible_city)

                if amount == 0:
                    print("AMOUNT E 0 !!!!!!!!!!!")

                weights.append(store[possible_city] / amount)

            if len(weights) != 0 and len(possible_city_list) != 0:
                next_city = choices(possible_city_list, weights=weights, k=1)[0]

        # add new city to the tour
        self.__tour.append(next_city)

        # encrease the length of tour

        self.__cost += self.__problemParam["distance_matrix"][self.__tour[-2]][
            self.__tour[-1]
        ]

        self.__pheromone_matrix[self.__tour[-2]][self.__tour[-1]] = 1
        self.__pheromone_matrix[self.__tour[-1]][self.__tour[-2]] = 1

