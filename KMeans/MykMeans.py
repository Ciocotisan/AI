import numpy as np
import sys
import math
from Cluster import Cluster


class kMeans:
    def __init__(self, clustersNumber, inputs):

        self.__clustersNumber = clustersNumber
        self.__clusters = []
        self.__inputs = inputs
        self.__isChange = True

    def runAlgo(self):
        self.initializeCentroids()

        while self.__isChange:

            for cluster in self.__clusters:
                cluster.objects = []

            for feat in self.__inputs:
                lowest_distance = sys.maxsize
                cluster_index = -1
                # put each feture in the closest cluster
                for i in range(len(self.__clusters)):
                    dist = self.euclidianDistance(self.__clusters[i].centroid, feat)
                    # dist = self.levenshtein_distance(self.__clusters[i].centroid, feat)
                    if dist < lowest_distance:
                        lowest_distance = dist
                        cluster_index = i

                self.__clusters[cluster_index].objects.append(feat)

            self.__isChange = False

            # update the centroid for each cluster
            for i in range(self.__clustersNumber):

                current_centroid = self.__clusters[i].centroid
                new_centroid = self.computeNewCentroidForClusterI(i)

                if (
                    self.checkCentroidsSimilarity(current_centroid, new_centroid)
                    == False
                ):
                    self.__isChange = True
                    self.__clusters[i].centroid = new_centroid

    def make_prediction(self, testInput):
        computedOutputs = []

        for feat in testInput:
            lowest_distance = sys.maxsize
            cluster_index = -1

            # put each feture in the closest cluster
            for i in range(len(self.__clusters)):
                dist = self.euclidianDistance(self.__clusters[i].centroid, feat)
                # dist = self.levenshtein_distance(self.__clusters[i].centroid, feat)

                if dist < lowest_distance:
                    lowest_distance = dist
                    cluster_index = i

            computedOutputs.append(cluster_index)

        return computedOutputs

    # function returs True is crtCentroid and newCentroid are the same
    #                False otherwise
    def checkCentroidsSimilarity(self, crtCentroid, newCentroid):
        for o, n in zip(crtCentroid, newCentroid):
            if o != n:
                return False

        return True

    def computeNewCentroidForClusterI(self, index):
        new_centroid = [0 for _ in range(len(self.__inputs[0]))]

        for feat in self.__clusters[index].objects:
            for j in range(len(feat)):
                new_centroid[j] += feat[j]

        for j in range(len(new_centroid)):
            new_centroid[j] /= len(self.__clusters[index].objects)

        return new_centroid

    def levenshtein_distance(self, centroid, feature):
        len_x = len(centroid) + 1
        len_y = len(feature) + 1
        matrix = [[0 for _ in range(len_y)] for _ in range(len_x)]

        for x in range(len_x):
            matrix[x][0] = x
        for y in range(len_y):
            matrix[0][y] = y

        for x in range(1, len_x):
            for y in range(1, len_y):
                if centroid[x - 1] == feature[y - 1]:
                    matrix[x][y] = min(
                        matrix[x - 1][y] + 1, matrix[x - 1][y - 1], matrix[x][y - 1] + 1
                    )
                else:
                    matrix[x][y] = min(
                        matrix[x - 1][y] + 1,
                        matrix[x - 1][y - 1] + 1,
                        matrix[x][y - 1] + 1,
                    )
        return matrix[len_x - 1][len_y - 1]

    def euclidianDistance(self, centroid, feature):
        sum = 0

        for c, f in zip(centroid, feature):
            sum += (c - f) ** 2

        return math.sqrt(sum)

    def initializeCentroids(self):
        indexes = [i for i in range(len(self.__inputs))]
        positions = np.random.choice(indexes, self.__clustersNumber, replace=False)

        for i in range(self.__clustersNumber):
            self.__clusters.append(Cluster(self.__inputs[positions[i]]))

    def dunnIndex(self):

        lowest_distance_intercluster = sys.maxsize

        for i in range(self.__clustersNumber - 1):
            for j in range(i + 1, self.__clustersNumber):
                dist = self.euclidianDistance(
                    self.__clusters[i].centroid, self.__clusters[j].centroid
                )
                if dist < lowest_distance_intercluster:
                    lowest_distance_intercluster = dist

        max_distance_intracluster = 0

        for cluster in self.__clusters:
            for i in range(len(cluster.objects) - 1):
                for j in range(i + 1, len(cluster.objects)):
                    dist = self.euclidianDistance(
                        cluster.objects[i], cluster.objects[j]
                    )
                    if dist > max_distance_intracluster:
                        max_distance_intracluster = dist

        return lowest_distance_intercluster / max_distance_intracluster

