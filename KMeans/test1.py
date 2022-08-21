import numpy as np


def levenshtein_distance(centroid, feature):
    len_x = len(centroid) + 1
    len_y = len(feature) + 1
    matrix = [[0 for _ in range(len_y)] for _ in range(len_x)]
    print(matrix)

    for x in range(len_x):
        print(x)
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
                    matrix[x - 1][y] + 1, matrix[x - 1][y - 1] + 1, matrix[x][y - 1] + 1
                )
    print(matrix)
    return matrix[len_x - 1][len_y - 1]


c = "cosmin"
a = "cosmins"

print(levenshtein_distance(c, a))

