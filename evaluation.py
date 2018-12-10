import os 
import numpy as np
import networkx as nx
from treeApproximation import TreeApproximator, ComTreeNode, create_tree_from_laminar_family
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
                "road": "roadNet-CA.txt"
                }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=str, default='real', help='[real, synthetic]')
    parser.add_argument('--dataset', type=str, help='Keyword for the dataset.')
    parser.add_argument('--alpha', type=str)
    parser.add_argument('--beta', type=str)
    parser.add_argument('--n', type=int)
    parser.add_argument('--m', type=int)
    parser.add_argument('--h', type=int)
    parser.add_argument('--r', type=int)
    parser.add_argument('--seed', type=int, default=6820)

    return args

#################################################################################
#						    Generate/ Load Graphs 								#
#################################################################################
def _get_synthetic():
	"""
	Return a list of synthetic graphs for each type of graphs.
	"""
    list_g = [graphs.random_graph(args.n),
              graphs.cycle_g(args.n),
              graphs.vertices(args.n),
              graphs.bipartite(args.n, args.m),
              graphs.grids(args.n, args.m),
              graphs.binarytree(args.h, args.r)]
    
    return list_g

def _get_real():
	"""
	Return the real dataset of choice as a graph.
	"""
    assert args.dataset in dataset_names.keys(), 'Invalid dataset keyword. '
    
    return load_graph(dataset_names[args.dataset])

def get_dataset():
    if args.evaluation == 'real':
        return [_get_real()] # Making it a singleton list to match the data type
    elif args.evaluation == 'synthetic':
        return _get_synthetic() 
    else:
        raise Exception('Evaluation type [--evaluation] must be [real] or [synthetic].')

#################################################################################
#			                       Verifier										#
#################################################################################








#################################################################################
#			                 Approximation Evaluation							#
#################################################################################




#################################################################################
#			                    Run-time Evaluation								#
#################################################################################
import time as t




#################################################################################
#						     	      Main										#
#################################################################################
if __name__ == '__main__':
    args = get_args()
    dataset = get_dataset()
















