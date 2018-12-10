import numpy as np
import networkx as nx
from treeApproximation import TreeApproximator
import matplotlib.pyplot as plt

# class T:
#     def __init__(self, G):
#         nodes = G.nodes()
#         edges = G.edges()
#         n = len(nodes)
#         m = len(edges)
#         self.table = self._create_table(n, edges)


#     def _create_table(self, n, edges):
#         table = np.zeros((n,n))
#         for edge, _ in edges.items():
#             source_id = edge[0]
#             target_id = edge[1]

#             table[source_id][target_id] = 1
#             table[target_id][source_id] = 1
#         return table

#     def produce_spanner(self):
#         h = nx.from_numpy_matrix(self.table)
#         return h

#     def __str__(self):
#         return str(self.table)


class Graph_Spanner:

    def __init__(self, graph, alpha=None, beta=None, spanner_func="greedy"):
        self.g = graph
        self.alpha = alpha
        self.beta = beta

        self.h = self.greedy_spanner(2)

    def distance_g(self, u, v):
        try:
            return nx.shortest_path_length(self.g, u, v)
        except nx.NetworkXNoPath as e:
            return None

    def distance_h(self, u, v):
        try:
            return nx.shortest_path_length(self.h, u, v)
        except nx.NetworkXNoPath as e:
            return None

    def greedy_spanner(self, k):
        g = self.g
        print("start")
        dists = nx.floyd_warshall_numpy(g).tolist()
        print("end")
        h = nx.Graph()
        for i in g.nodes():
            for j in g.nodes():
                d = dists[i][j]
                if (d is not None) and (d > 2 * k - 1):
                    h.add_edge(i, j)
        return h

    def m_H(self):
        """
        m(H): size (# of edges) of spanner graph.
        Lower is better. Performance metric. Returns an int.
        """
        return self.h.number_of_edges()



import graphs as localGraphs
import sys

# print("loading email_graph...")
# email_graph = localGraphs.email_graph()
# print("successfully loaded email_graph!")
# print("Creating spanner")
# email_spanner = Graph_Spanner(email_graph)
# print("\n")

# print("loading road_graph...")
# road_graph =  localGraphs.road_graph()
# print("successfully loaded road_graph!")
# road_spanner = Graph_Spanner(road_graph)
# print("\n")

print("loading collab graph...")
msg_graph = localGraphs.msg_graph()
print("successfully loaded collab_graph")
msg_graph = Graph_Spanner(msg_graph)
print("\n")

