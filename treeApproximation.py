import numpy as np
import networkx as nx
import random
from typing import Dict, Optional, List
import matplotlib.pyplot as plt

def _beta_bad():
    l_2 = np.log2(2)
    l_2_inv = 1 / l_2
    x = np.random.random_sample()
    x = l_2_inv * x + l_2_inv
    x = 1 / (x * l_2)
    return x + 1
delta = 0.01
intervals = np.arange(1, 2, delta)
pdf = np.log(intervals+delta) - np.log(intervals)
pdf /= np.log(2)

def _beta():
    return np.random.choice(intervals, p=pdf)

def _beta_2():
    l_2 = np.log(2)
    x = np.random.random_sample()
    x = np.exp(x*l_2)
    return x

class ComTreeNode:
    def __init__(self, elems, beta, children=None):
        self.elems = elems
        self.beta = beta
        self.children: Optional[List[ComTreeNode]] = children

    def is_leaf(self) -> bool:
        return self.children is None

    def _to_nx_graph_helper(self, orig_dists, aproxG: nx.Graph, c: List[int], parent: Optional[int], parent_diam):
        this_node = c[0]
        if len(self.elems) == 1:
            this_node = self.elems[0]
        else:
            c[0] += 1

        diam = max(map(lambda dic: max(map(lambda e: dic[e], self.elems)), map(lambda e: orig_dists[e], self.elems)))
        aproxG.add_node(this_node, elems=self.elems, diam=diam) # add diameter for sub graphs

        if parent is not None:
            aproxG.add_edge(this_node, parent, dist = parent_diam)
        if self.children is not None:
            for child in self.children:
                child._to_nx_graph_helper(orig_dists, aproxG, c, this_node, diam)

    def to_nx_graph(self, n, orig_dists, g):
        c = [n]
        if len(g.nodes()) > 0:
            c[0] = max(c[0], max(g.nodes()) + 1)
        self._to_nx_graph_helper(orig_dists, g, c, None, None)
        return g


def _create_tree_helper(laminar_family, betas, set, i) -> List[ComTreeNode]:
    ret = []
    j = 0
    if i == len(laminar_family) - 1:
        while j < len(laminar_family[i]):
            if len(laminar_family[i][j]) > 0 and laminar_family[i][j][0] in set:
                ret.append(ComTreeNode(laminar_family[i][j], betas[i]))
            j += 1
    else:
        ret = []
        while j < len(laminar_family[i]):
            if len(laminar_family[i][j]) > 0 and laminar_family[i][j][0] in set:
                new_nodes = _create_tree_helper(laminar_family, betas, laminar_family[i][j], i + 1)
                ret.append(ComTreeNode(laminar_family[i][j], betas[i], new_nodes))
            j += 1
    return ret


def create_tree_from_laminar_family(laminar_family, betas) -> ComTreeNode:
    return ComTreeNode(laminar_family[0][0], betas[0], _create_tree_helper(laminar_family, betas, laminar_family[0][0], 1))


class TreeApproximator(object):
    def __init__(self, G: nx.Graph):
        G = G.to_undirected()
        connected_components = list(map(G.subgraph, sorted(nx.connected_components(G), key=len, reverse=True)))
        self.node_dists = {}

        g = nx.Graph()
        for comp in connected_components:
            self.G = comp
            if len(comp.nodes()) == 1:
                node = list(comp.nodes())[0]
                g.add_node(node, elems=[node], diam=0)
            else:
                g = self._create_spanning_tree_approx().to_nx_graph(len(G.nodes()),self.node_dists, g)

        self.spanning_tree_approx: nx.Graph = g

    def _distance_dict(self, node_list) -> Dict[int, Dict[int, float]]:
        dict : Dict[int, Dict[int, int]] = {}
        dists = nx.floyd_warshall_numpy(self.G, node_list).tolist()

        for i in range(0, len(node_list)):
            if node_list[i] not in dict:
                dict[node_list[i]] = {}
            dict[node_list[i]][node_list[i]] = 0.0
            for j in range(0, i):
                if node_list[j] not in dict:
                    dict[node_list[j]] = {}
                dict[node_list[i]][node_list[j]] = dists[i][j]
                dict[node_list[j]][node_list[i]] = dists[j][i]
        return dict

    def _create_spanning_tree_approx(self) -> ComTreeNode:
        pi = list(self.G.nodes())
        random.shuffle(pi)
        self.node_dists = self._distance_dict(pi)

        beta = _beta()
        # print("Beta: %s" % (beta,))
        diameter = max(map(lambda x: max(x.values()), self.node_dists.values()))
        # print(diameter)
        delta = np.log2(diameter)
        delta = int(delta) + 1

        D = [[] for _ in range(0, delta+1)]
        betas = []

        i = delta-1
        D[i+1] = [pi]
        betas.append(np.power(2.0, delta) * beta)
        while max(map(len, D[i+1])) > 1:
            beta_i = np.power(2.0, i-1) * beta
            betas.append(beta_i)
            nodes = []

            for l in range(0, len(pi)):
                for cluster in D[i+1]:
                    if cluster != [] :
                        append_nodes = filter(lambda x: x not in nodes, cluster)
                        append_nodes = list(append_nodes)
                        append_nodes = filter(lambda x: self.node_dists[x][pi[l]] < beta_i, append_nodes)
                        append_nodes = list(append_nodes)
                        nodes.extend(append_nodes)
                        D[i].append(append_nodes)

            i -= 1



        D = D[i+1:]
        D.reverse()
        # print("Depth of tree %s" % (len(D)))
        return create_tree_from_laminar_family(D, betas)

    def _get_approx_dist(self, a, b):
        head = self.spanning_tree_aprox
        min_beta = head.beta
        while head is not None:
            for child in head.children:
                if a in child.elems and b in child.elems:
                    min_beta = child.beta
                    if not child.is_leaf():
                        head = child
                    else:
                        head = None
        return min_beta



if __name__ == "__main__":
    # laminar_families = [
    #     [['a', 'b', 'c']],
    #     [['a'], ['b', 'c']],
    #     [['a'], ['b'], ['c']]
    # ]
    # betas = [
    #     1,
    #     2,
    #     3
    # ]
    #
    # tree = create_tree_from_laminar_family(laminar_families, betas)
    # print(tree)

    # g = nx.cycle_graph(10)
    # nx.draw_networkx(g)

    # approx = TreeApproximator(g)

    print(_beta())
