import math
from abc import ABC, abstractmethod


class LinkPrediction(ABC):
    def __init__(self, graph):
        self.graph = graph
        self.adj = None
        self.deg = None

    def neighbors(self, v):
        return list(self.graph.neighbors(v))

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Fit must be implemented")

    @abstractmethod
    def score(self, u, v):
        raise NotImplementedError("Score must be implemented")


class CommonNeighbors(LinkPrediction):
    def fit(self):
        self.adj = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes()}
        self.deg = {n: len(self.adj[n]) for n in self.adj}
        return self

    def score(self, u, v):
        return float(len(self.adj[u] & self.adj[v]))


class Jaccard(LinkPrediction):
    def fit(self):
        self.adj = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes()}
        self.deg = {n: len(self.adj[n]) for n in self.adj}
        return self

    def score(self, u, v):
        inter = len(self.adj[u] & self.adj[v])
        union = self.deg[u] + self.deg[v] - inter
        if union == 0:
            return 0.0
        return float(inter) / float(union)


class AdamicAdar(LinkPrediction):
    def fit(self):
        self.adj = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes()}
        self.deg = {n: len(self.adj[n]) for n in self.adj}
        return self

    def score(self, u, v):
        inter = self.adj[u] & self.adj[v]
        s = 0.0
        for w in inter:
            dw = self.deg[w]
            if dw > 1:
                s += 1.0 / math.log(dw)
        return float(s)
