import numpy as np
import networkx as nx
from treeApproximation import TreeApproximator
import matplotlib.pyplot as plt
import graphs as localGraphs
import sys
import tqdm
import dynamic_distance


class Graph_Spanner:

    def __init__(self, graph, alpha=None, beta=None, k=2, spanner_func="greedy"):
        self.g = graph
        self.alpha = alpha
        self.beta = beta
        self.k = k

        self.h = self.greedy_spanner()
        print("k: %s" % k)

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

    def greedy_spanner(self):
        g = self.g.to_undirected()
        connected_components = list(map(g.subgraph, sorted(nx.connected_components(g), key=len, reverse=True)))

        k = self.k
        h = nx.Graph()
        dists = {}
        # h_dists = dynamic_distance.DynamicDistance(h)

        total = map(len, connected_components)
        total = map(lambda x: x**2, total)
        total = sum(total)

        pbar = tqdm.tqdm(total=total)

        for component in connected_components:

            if len(component.nodes()) == 1:
                node = list(component.nodes())[0]
                h.add_node(node, elems=[node], diam=0)

            else:
                for i in component.nodes():
                    h.add_node(i)

                for (i, j) in component.edges():

                    try:
                        d = nx.shortest_path_length(h, i, j)
                    except nx.NetworkXNoPath as e:
                        d = np.inf
                    if d > 2 * k - 1:
                            h.add_edge(i, j)

                # for i in component.nodes():
                #     for j in component.nodes():
                #         pbar.update(1)
                #         if not component.has_edge(i, j):
                #             continue

                #         try:
                #             d = nx.shortest_path_length(h, i, j)
                #         except nx.NetworkXNoPath as e:
                #             d = np.inf

                #         # d = h_dists.get_distance(i, j)
                #         if d > 2 * k - 1:
                #             h.add_edge(i, j)
        pbar.close()
        return h

    def m_H(self):
        """
        m(H): size (# of edges) of spanner graph.
        Lower is better. Performance metric. Returns an int.
        """
        return self.h.number_of_edges()


if __name__ == "__main__":

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

    print("loading cycle graph...")
    cycle_graph = localGraphs.cycle(10)
    print("Successfully loaded cycle graph")
    cycle_spanner = Graph_Spanner(cycle_graph, k=3)
    print(cycle_spanner.h.nodes())
    print(cycle_spanner.h.edges())
    print("Finished creating spanner for cycle graph")


    # print("loading msg graph...")
    # msg_graph = localGraphs.msg_graph()
    # print("successfully loaded msg graph!")
    # msg_spanner = Graph_Spanner(msg_graph)
    # print("Finished creating spanner for msg graph")
    # print("\n")

