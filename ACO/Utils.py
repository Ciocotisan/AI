import math


class Utils:
    def __init__(self, inputFileName, outputFileName="files\\output.txt"):
        self.__input = inputFileName
        self.__output = outputFileName
        self.__n = -1
        self.__matrix = []

    def get_nrV(self):
        return self.__n

    def get_matrix(self):
        return self.__matrix

    def writeToFile(self, n, road):
        f = open(self.__output, "w")

        f.write(str(n) + "\n")

        for el in road:
            f.write(str(el) + " ")

        f.close()

    def readFromFileMatrix(self):
        f = open(self.__input, "r")

        self.__n = int(f.readline().strip())

        for _ in range(self.__n):
            line = f.readline().strip()

            values = line.split(",")

            lst = []

            for el in values:
                lst.append(float(el))

            self.__matrix.append(lst)

        f.close()

    def readFromFileCoordonates(self):

        f = open(self.__input, "r")
        lines = f.readlines()
        f.close()

        coordonate = []

        for line in lines:

            line.strip()
            args = line.split(" ")
            if len(args) > 0:
                x = float(args[1])
                y = float(args[2])
                coordonate.append([x, y])

        n = len(coordonate)
        self.__n = n

        for _ in range(n):
            self.__matrix.append([0 for _ in range(n)])

        for i in range(n):
            for j in range(n):
                if i < j:

                    x1 = coordonate[i][0]
                    y1 = coordonate[i][1]

                    x2 = coordonate[j][0]
                    y2 = coordonate[j][1]

                    dist = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

                    if dist == 0:
                        print("INPUT GRESIT ARE 2 noduri pe ACEEASI POZITIE")

                    self.__matrix[i][j] = dist
                    self.__matrix[j][i] = dist

