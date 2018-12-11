import os
import numpy as np
import networkx as nx
from treeApproximation import TreeApproximator, ComTreeNode, create_tree_from_laminar_family
from spanner import Graph_Spanner
import graphs
import matplotlib.pyplot as plt
import argparse


#################################################################################
#						    Parser and Preprocess 								#
#################################################################################

dataset_names = {
                # 1,005   25,571  http://snap.stanford.edu/data/email-Eu-core.html
                "email" : "email-Eu-core.txt",

                # 1,899  20,296   http://snap.stanford.edu/data/CollegeMsg.html
                "msg" : "CollegeMsg.txt",

                # 5,242 14,496    http://snap.stanford.edu/data/ca-GrQc.html
                "collab" : "ca-GrQc.txt",

                # 6,301   20,777   http://snap.stanford.edu/data/p2p-Gnutella08.html
                "p2p" : "p2p-Gnutella08.txt",

                # 1,965,206 2,766,607   http://snap.stanford.edu/data/roadNet-CA.html
                "road" : "roadNet-CA.txt"
                }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=str, default='real', help='[real, synthetic]')
    parser.add_argument('--dataset', type=str, help='Keyword for the dataset.')
    parser.add_argument('--mode', type=str, help='[tree] or [spanner]')
    parser.add_argument('--alpha', type=str)
    parser.add_argument('--beta', type=str)
    parser.add_argument('--n', type=int)
    parser.add_argument('--m', type=int)
    parser.add_argument('--h', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--r', type=int)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--seed', type=int, default=6820)


    return parser.parse_args()

#################################################################################
#						    Generate/ Load Graphs 								#
#################################################################################
def _get_synthetic():
    """
    Return a dict of synthetic graphs for each type of graphs.
    """
    dict_g = {
        'random'     : graphs.random_graph(args.n),
        'cycle'      : graphs.cycle(args.n),
        'vertices'   : graphs.vertices(args.n),
        'bipartite'  : graphs.bipartite(args.n, args.m),
        'grids'      : graphs.grids(args.n, args.m),
        'binarytree' : graphs.binarytree(args.h, args.r)
    }

    return dict_g

def _get_real():
    """
    Return the real dataset of choice as a graph.
    """
    assert args.dataset in dataset_names.keys(), 'Invalid dataset keyword. '

    return {args.dataset : load_graph(dataset_names[args.dataset])}

def get_graphs():
    if args.evaluation == 'real':
        return _get_real() # Making it a singleton list to match the data type
    elif args.evaluation == 'synthetic':
        return _get_synthetic()
    else:
        raise Exception('Invalid evaluation type [--evaluation] must be [real] or [synthetic].')


#################################################################################
#			                 Approximation Evaluation							#
#################################################################################

def approximation_rate(g, hs, mode = "max"):
    # avaerge across hs
    n = len(g.nodes())
    original = nx.floyd_warshall_numpy(g).A
    d_approx = np.array(list(map(lambda h: nx.floyd_warshall_numpy(h).A[:n,:n], hs))).mean(axis = 0)
    if mode == "max":
        return np.nan_to_num(d_approx / original).max()
    else:
        return np.nan_to_num(d_approx / original).mean()


def evaluate_approx(input_graphs):
    result_dict = {}
    for key_g, g in input_graphs.items():
        if args.mode == 'tree':
            hs = [TreeApproximator(g).spanning_tree_approx for _ in range(args.repeats)]
        if args.mode == 'spanner':
            hs = [Graph_Spanner(g, args.alpha, args.beta, args.k).h]
        ratio = approximation_rate(g, hs)
        print(key_g)
        print(ratio)
        result_dict[key_g] = ratio

    return result_dict


#################################################################################
#			                    Run-time Evaluation								#
#################################################################################
import time as t
def evaluate_runtime():
    # Use loglog plotting
    time_result = []
    for i in range(args.repeats):
        n = 10 ** (i+1)
        time_i = []
        for _ in range(args.repeats):
            g = graphs.random_graph(n)
            t1 = t.time()
            if args.mode == 'tree':
                _ = TreeApproximator(g).spanning_tree_approx
            elif args.mode == 'spanner':
                _ = Graph_Spanner(g, args.alpha, args.beta, args.k).h
            else:
                raise Exception('Invalid mode. ')
            t2 = t.time()
            time_i.append(t2-t1)
        time_result.append([n, np.mean(time_i)])

    return np.array(time_result)


#################################################################################
#                                    Plotter                                    #
#################################################################################

def plot_eval():
    raise Exception('Not done yet. ')


#################################################################################
#						     	      Main										#
#################################################################################
if __name__ == '__main__':
    args = get_args()
    input_graphs = get_graphs()
    eval_approx = evaluate_approx(input_graphs)
    np.save("apprx_result.npy", eval_approx)
    # eval_runtime = evaluate_runtime()
    np.save("eval_result.npy", eval_runtime)
    # plot_eval()



















