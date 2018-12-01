import numpy as np
import networkx as nx
import random
from typing import Dict, Optional, List


def _beta():
    l_2 = np.log2(2)
    l_2_inv = 1 / l_2
    x = np.random.random_sample()
    x = l_2_inv * x + l_2_inv
    x = 1 / (x * l_2)
    return x


class ComTreeNode:
    def __init__(self, elems, beta, children=None):
        self.elems = elems
        self.beta = beta
        self.children: Optional[List[ComTreeNode]] = children

    def is_leaf(self) -> bool:
        return self.children is None

    def _to_nx_graph_helper(self, g: nx.Graph, c: List[int], parent: Optional[int]):
        this_node = c[0]
        g.add_node(c[0], elems=self.elems) # add diameter for sub graphs
        c[0] += 1
        if parent is not None:
            g.add_edge(this_node, parent)
        if self.children is not None:
            for child in self.children:
                child._to_nx_graph_helper(g, c, this_node)

    def to_nx_graph(self):
        g = nx.Graph()
        c = [0]
        self._to_nx_graph_helper(g, c, None)
        return g


def _create_tree_helper(counter, laminar_family, betas, set, i) -> List[ComTreeNode]:
    ret = []
    if i == len(laminar_family) - 1:
        while counter[i] < len(laminar_family[i]) and laminar_family[i][counter[i]][0] in set:
            ret.append(ComTreeNode(laminar_family[i][counter[i]], betas[i]))
            counter[i] += 1
    else:
        ret = []
        while counter[i] < len(laminar_family[i]) and laminar_family[i][counter[i]][0] in set:
            new_nodes = _create_tree_helper(counter, laminar_family, betas, laminar_family[i][counter[i]], i + 1)
            ret.append(ComTreeNode(laminar_family[i][counter[i]], betas[i], new_nodes))
            counter[i] += 1
    return ret


def create_tree_from_laminar_family(laminar_family, betas) -> ComTreeNode:
    counter = [0 for _ in range(len(laminar_family))]
    return _create_tree_helper(counter, laminar_family, betas, laminar_family[-1][0], 0)[0]


class TreeApproximator(object):
    def __init__(self, G: nx.Graph):
        self.G = G
        self.spanning_tree_aprox: nx.Graph = self._create_spanning_tree_approx().to_nx_graph()

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
        node_dists = self._distance_dict(pi)

        beta = _beta()
        diameter = max(map(lambda x: max(x.values()), node_dists.values()))
        print(diameter)
        delta = np.log2(diameter)
        delta = int(delta)

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
                    append_nodes = filter(lambda x: x not in nodes, cluster)
                    append_nodes = list(append_nodes)
                    append_nodes = filter(lambda x: node_dists[x][pi[l]] < beta_i, append_nodes)
                    append_nodes = list(append_nodes)
                    nodes.extend(append_nodes)

                    D[i].append(append_nodes)

            i -= 1
        D = D[i+1:]
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

    g = nx.cycle_graph(10)
    # nx.draw_networkx(g)

    approx = TreeApproximator(g)
