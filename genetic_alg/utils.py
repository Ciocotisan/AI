from random import randint


def readFile(edges):

    f = open("files/file.in", "r")

    lines = f.readlines()

    id = -1
    s = -1
    d = -1

    for i in range(len(lines)):
        line = lines[i].strip()

        if "id" in line:
            args = line.split()
            id = int(args[1])

        if "source" in line:
            args = line.split()
            s = int(args[1])

        if "target" in line:
            args = line.split()
            d = int(args[1])
            edges.append([s, d])

    return id


def get_adjacent_list(adj_list):
    edges = []
    n = readFile(edges)

    for _ in range(n + 1):
        adj_list.append([])

    for edge in edges:
        adj_list[edge[0]].append(edge[1])
        adj_list[edge[1]].append(edge[0])

    return n


def readFileMatrix():
    f = open("files/my_input_1.txt")

    n = int(f.readline())
    print(n)

    mat = [[] for _ in range(n)]

    for i in range(n):
        line = f.readline()
        args = line.split(", ")

        for j in range(n):
            mat[i].append(int(args[j]))

    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if mat[i][j] == 1:
                edges.append([i, j])

    print(edges)


readFileMatrix()


def generateNewValues(adj_list):
    v = [0 for i in range(len(adj_list))]

    for i in range(len(adj_list)):
        poz = randint(0, len(adj_list[i]) - 1)
        v[i] = adj_list[i][poz]

    return v


def dfs(x, adj_list, c, index):
    c[x] = index

    for vecin in adj_list[x]:
        if c[vecin] == -1:
            dfs(vecin, adj_list, c, index)


def devideChromosomeInCommunities(chromosome):

    lg = len(chromosome)
    adj_list = [[] for _ in range(lg)]

    for i in range(lg):
        adj_list[i].append(chromosome[i])
        adj_list[chromosome[i]].append(i)

    index = 1
    c = [-1 for _ in range(lg)]

    for i in range(lg):
        if c[i] == -1:
            dfs(i, adj_list, c, index)
            index += 1

    return c


def fitness(chromosome, adj_list):

    lg = len(chromosome)
    k = [0 for _ in range(lg)]

    compConexe = devideChromosomeInCommunities(chromosome)
    s = 0

    for i in range(lg):
        k[i] = len(adj_list[i])
        s += k[i]

    m = s // 2

    sum = 0

    for i in range(lg):
        for j in range(i + 1, lg):
            c_aux = 0

            if compConexe[i] == compConexe[j]:
                c_aux = 1

            if j in adj_list[i]:
                sum += (1 - (k[i] * k[j]) // 2 * m) * c_aux
            else:
                sum += (0 - (k[i] * k[j]) // 2 * m) * c_aux

    rez = 1 // 2 * m + sum

    return rez
