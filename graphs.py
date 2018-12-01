import numpy as np
import networkx as nx
from treeApproximation import TreeApproximator
import matplotlib.pyplot as plt

def random_graph(n):
    m = np.random.rand(n,n) > 0.5
    return nx.from_numpy_matrix(m)


def cycle(n):
    return nx.cycle_graph(n)


def eamil_graph():
    dg = nx.DiGraph()
    with open("data/email-Eu-core.txt", "r") as f:
        content = f.readlines()
        for line in content:
            vertices = line.strip().split(" ")
            a = int(vertices[0])
            b = int(vertices[1])

            dg.add_edge(a,b, weight=1)
    return dg


def visualize(g):
    plt.figure()
    nx.draw_networkx(g)
    plt.show()


if __name__ == "__main__":

    visualize(eamil_graph())

