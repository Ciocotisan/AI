class Neuron:
    def __init__(self, w=[], out=None, delta=0.0):
        self.weights = w  # tin minte doar weight'urile care intra in neuron
        self.output = out  # valoare calculata de neuron
        self.delta = delta  # eroarea pe care trebuie sa o intoarca
        self.prob = None  # probabilitatea ptr un nod de output

    def __str__(self):
        return (
            "weights: "
            + str(self.weights)
            + ", output: "
            + str(self.output)
            + ", delta: "
            + str(self.delta)
        )

    def __repr__(self):
        return (
            "weights: "
            + str(self.weights)
            + ", output: "
            + str(self.output)
            + ", delta: "
            + str(self.delta)
        )

