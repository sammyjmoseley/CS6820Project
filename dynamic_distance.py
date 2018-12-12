import networkx as nx
import numpy as np
class DynamicDistance(object):
    def __init__(self, g):
        self.g = g
        self.distances = {}

    def _calculated_distances(self):
        pass # TODO

    def _update_distance(self, nodes_changed, n1, n2):
        changed = False
        for s1 in self.distances[n1].keys():
            d = self.distances[n1][s1] + 1
            if s1 not in self.distances[n2] or self.distances[n2][s1] > d:
                self.distances[n2][s1] = d
                self.distances[s1][n2] = d
                changed = True

        if changed:
            nodes_changed.append(n2)

    def add_edge(self, a, b):
        self.g.add_edge(a, b)
        if a not in self.distances:
            self.distances[a] = {}

        if b not in self.distances:
            self.distances[b] = {}
        
        self.distances[a][b] = 1
        self.distances[b][a] = 1
        nodes_visited = [a, b]
        nodes_changed = [a, b]
        while len(nodes_changed) > 0:
            n1 = nodes_changed.pop()
            if n1 in nodes_visited:
                continue
            nodes_visited.append(n1)
            neighbors = self.g.neighbors(n1)
            for n2 in neighbors:
                if n2 not in nodes_visited:
                    self._update_distance(nodes_changed, n1, n2)

    def get_distance(self, n1, n2):
        if n1 in self.distances and n2 in self.distances[n1]:
            return self.distances[n1][n2]
        return np.inf
