import numpy as np
import networkx as nx
from networkx.algorithms.bipartite import generators
from treeApproximation import TreeApproximator, ComTreeNode, create_tree_from_laminar_family
import matplotlib.pyplot as plt
import sys


def random_graph(n):
    m = np.random.rand(n,n) > 0.5
    return nx.from_numpy_matrix(m)


def cycle(n):
    return nx.cycle_graph(n)


def vertices(n):
    return nx.from_numpy_matrix(np.zeros([n,n]))


def bipartite(n, m=None):
    if not m:
        m = n
    return generators.random_graph(n, m, 0.2)


def grids(n,m=None):
    if not m:
        m = n
    def mapping(x):
        return x[0]*m + x[1]
    return nx.relabel_nodes(nx.grid_graph([n,m]), mapping)


def binarytree(h, r = 2):
    return nx.balanced_tree(r,h)


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
    # 1,005   25,571  http://snap.stanford.edu/data/email-Eu-core.html
    return load_graph("email-Eu-core.txt")


def msg_graph():
    # 1,899  20,296   http://snap.stanford.edu/data/CollegeMsg.html
    return load_graph("CollegeMsg.txt")


def collab_graph():
    # 5,242 14,496    http://snap.stanford.edu/data/ca-GrQc.html
    return load_graph("ca-GrQc.txt")


def p2p_graph():
    # 6,301   20,777   http://snap.stanford.edu/data/p2p-Gnutella08.html
    return load_graph("p2p-Gnutella08.txt")

def road_graph():
    # 1,965,206 2,766,607   http://snap.stanford.edu/data/roadNet-CA.html
    return load_graph("roadNet-CA.txt")


def visualize(g, labels = None):
    plt.figure()
    pos = nx.spring_layout(G = g, dim = 2, k = 10, scale=20)
    nx.draw_networkx(g, pos)
    if labels != None:
      nx.draw_networkx_edge_labels(g, pos, labels)


if __name__ == "__main__":

    g = email_graph()

    visualize(g)
    g_ = TreeApproximator(g).spanning_tree_approx
    dic = {}
    for a, b, data in g_.edges(data = True):
        dic[(a,b)]= data['dist']
    visualize(g_, labels = dic)
    print(len(g_.nodes()))
    plt.show()



