import numpy as np
import networkx as nx
from treeApproximation import TreeApproximator
import matplotlib.pyplot as plt

def random_graph(n):
    m = np.random.rand(n,n) > 0.5
    return nx.from_numpy_matrix(m)


def cycle(n):
    return nx.cycle_graph(n)


def load_graph(name):
    dg = nx.DiGraph()
    with open("data/"+name, "r") as f:
        content = f.readlines()
        for line in content:
            if line[0] == "#":
                continue
            vertices = line.split()
            a = int(vertices[0])
            b = int(vertices[1])
            dg.add_edge(a,b, weight=1)
    return dg


def email_graph():
    return load_graph("email-Eu-core.txt")

def road_graph():
    return load_graph("roadNet-CA.txt")

def collab_graph():
    return load_graph("ca-HepTh.txt")


def visualize(g):
    plt.figure()
    nx.draw_networkx(g)
    plt.show()


if __name__ == "__main__":
    visualize(email_graph())

