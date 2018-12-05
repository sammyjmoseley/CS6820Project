import numpy as np
import networkx as nx
from treeApproximation import TreeApproximator, ComTreeNode, create_tree_from_laminar_family
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


def visualize(g, labels = None):
    plt.figure()
    pos = nx.spring_layout(G = g, dim = 2, k = 10, scale=20)
    nx.draw_networkx(g, pos)
    if labels != None:
      nx.draw_networkx_edge_labels(g, pos, labels)


if __name__ == "__main__":

    # edge label example
    g = random_graph(10)
    visualize(g)
    dic = dict(zip(g.edges(), [1] * len(g.edges())))
    visualize(g, labels = dic)

    # tree approx example
    #g = email_graph()
    #visualize(g)
    #g_ = TreeApproximator(g).spanning_tree_aprox
    #visualize(g_)
    plt.show()



