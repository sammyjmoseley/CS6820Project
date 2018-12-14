import numpy as np
import networkx as nx
from treeApproximation import TreeApproximator, ComTreeNode, create_tree_from_laminar_family
from spanner import Graph_Spanner
import graphs

import os
import argparse
import logging
import datetime

#################################################################################
#						    Parser and Auxilary Functions 						#
#################################################################################

real_dataset_names = {
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

synthetic_dataset_functions = {
                            'random'     : graphs.random_graph,
                            'cycle'      : graphs.cycle,
                            'vertices'   : graphs.vertices, 
                            'bipartite'  : graphs.bipartite,
                            'binarytree' : graphs.binarytree,
                            'grids'      : graphs.grids,
                            }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ev', dest='evaluation', type=str, default='real', help='[real, synthetic]')
    parser.add_argument('--dl', dest='dataset_list', type=str, nargs='+', help='List of keywords of the datasets.')
    parser.add_argument('--et', dest='eval_type', type=str, default='runtime', help='[runtime] or [approx].')
    parser.add_argument('--md', dest='mode', type=str, help='[tree] or [spanner]')
    # parser.add_argument('--a', dest='alpha', type=str)
    # parser.add_argument('--b', dest='beta', type=str)
    parser.add_argument('--n', type=int)
    # parser.add_argument('--m', type=int)
    # parser.add_argument('--h', type=int)
    parser.add_argument('--k', type=int, default=2)
    # parser.add_argument('--r', type=int)
    parser.add_argument('--r', dest='repeats', type=int, default=10)
    parser.add_argument('--s', dest='seed', type=int, default=6820)
    # parser.add_argument('--am', dest='approx_mode', type=str, default='max', help='[max] or [avg]')
    parser.add_argument('--fa', dest='factor_analysis', type=str, default='False')

    return parser.parse_args()

def set_logger(log_path, log_name):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_dir = os.path.join(log_path, log_name)


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_dir)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

#################################################################################
#						    Generate/ Load Graphs 								#
#################################################################################
def _get_synthetic():
    """
    Return a dict of synthetic graphs for each type of graphs.
    """
    for d in args.dataset_list: assert d in synthetic_dataset_functions.keys(), 'Invalid dataset keyword. '

    input_map = {
                'random'     : (args.n,),
                'cycle'      : (args.n,),
                'vertices'   : (args.n,),
                'bipartite'  : (args.n//2, args.n//2),
                'grids'      : (int(np.sqrt(args.n)), int(np.sqrt(args.n))),
                'binarytree' : (int(np.log2(args.n)-1),) # When binary, the number of nodes is [2^height-1].
                }

    return {d : synthetic_dataset_functions[d](*input_map[d]) for d in args.dataset_list}

def _get_real():
    """
    Return the real dataset of choice as a graph.
    """
    for d in args.dataset_list: assert d in real_dataset_names.keys(), 'Invalid dataset keyword. '

    return {d : graphs.load_graph(real_dataset_names[d]) for d in args.dataset_list}

def get_graphs():
    if args.evaluation == 'real':
        return _get_real() 
    elif args.evaluation == 'synthetic':
        return _get_synthetic()
    else:
        raise Exception('Invalid evaluation type [--evaluation] must be [real] or [synthetic].')


#################################################################################
#			                 Approximation Evaluation							#
#################################################################################


def approximation_rate(g, hs, weight = None):
    # avaerge across hs
    n = len(g.nodes())
    # print(len(hs[0].nodes()))
    original = nx.floyd_warshall_numpy(g.to_undirected()).A
    # d_approx = np.array(list(map(lambda h: nx.floyd_warshall_numpy(h, nodelist=range(len(h.nodes())), weight=weight).A[:n,:n], hs)))
    # d_approx = d_approx.mean(axis = 0)
    d_approx = nx.floyd_warshall_numpy(hs, nodelist=range(len(hs.nodes())), weight=weight).A
    # print(d_approx)
    result = [[1 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if original[i][j] == 0:
                if d_approx[i][j] != 0:
                    result[i][j] = np.inf
                    logging.info(str(i) + " and " + str(j) + " connected -> disconnected.")
            elif original[i][j] == np.inf:
                if d_approx[i][j] != np.inf:
                    result[i][j] = np.inf
                    logging.info(str(i) + " and " + str(j) + " disconnected -> connected.")
            else:
                result[i][j] = d_approx[i][j] / original[i][j]
    assert np.nan_to_num(result).min() >= 1, "The minimum distortion should be greater than 1: " 

    return np.nan_to_num(result)


def evaluate_approx(input_graphs):
    result_dict = {}
    for key_g, g in input_graphs.items():
        max_list = []
        avg_list = []
        for idx in range(args.repeats):
            weight = None
            if args.mode == 'tree':
                hs = TreeApproximator(g).spanning_tree_approx
                weight = 'dist'
            if args.mode == 'spanner':
                hs = Graph_Spanner(g, k=args.k).h
            ratio_list = approximation_rate(g, hs, weight = weight)
            max_list.append(ratio_list.max())
            avg_list.append(ratio_list.mean())
        max_list = np.array(max_list)
        avg_list = np.array(avg_list)
        logging.info("Maximum distortions for {} graph: {:2.4f} +/- {:2.4f}.".format(key_g, max_list.mean(), max_list.std()))
        logging.info("Mean distortions for {} graph: {:2.4f} +/- {:2.4f}.\n".format(key_g, avg_list.mean(), avg_list.std()))
        result_dict[key_g] = (len(g.nodes()), (max_list, avg_list))

    return result_dict


#################################################################################
#			                    Run-time Evaluation								#
#################################################################################
import time as t
def evaluate_runtime():
    """
    out: 
        result_dict(dict): name of graph and the plot pair, where plot is nested tuple of (n, (mean, std)).
    """
    result_dict = {}
    base = 5 # For better evaluation, we add base number of nodes. 
    for key_g in args.dataset_list:
        test_graph = synthetic_dataset_functions[key_g]
        time_result = []
        full_result = []
        for i in range(args.n):
            n = 2 ** (i+1) + base

            if key_g == 'bipartite':
                n = n//2 
            elif key_g == 'grids':
                n = int(np.sqrt(args.n))
            elif key_g == 'binarytree': 
                n = int(np.log2(args.n)-1)

            logging.info('Runtime evaluation with N = {}.'.format(n))
            time_i = []
            for _ in range(args.repeats):
                g = test_graph(n)
                t1 = t.time()
                if args.mode == 'tree':
                    _ = TreeApproximator(g).spanning_tree_approx
                elif args.mode == 'spanner':
                    _ = Graph_Spanner(g, k=args.k).h
                else:
                    raise Exception('Invalid mode. ')
                t2 = t.time()
                time_i.append(t2-t1)
            logging.info('Runtime evaluation of {} graph with N={} complete. (time: {}.)\n'.format(key_g, n, str(datetime.timedelta(seconds=np.sum(time_i)))))
            time_result.append((n, np.mean(time_i)))
            full_result.append(np.array([[n]*args.repeats, time_i]))
            
        result_dict[key_g] = (time_result, full_result)

    return result_dict


#################################################################################
#                                 Visualization                                 #
#################################################################################
import matplotlib.pyplot as plt
def _plot_approx(result_dict):
    """
    Create box plot of approximation rates the algorithm of choice produced. 
    """
    label = []
    maxplots = []
    avgplots = []
    for key_g, value_g in result_dict.items():
        _, (max_list, avg_list) = value_g
        label.append(key_g)
        maxplots.append(max_list)
        avgplots.append(avg_list)
    
    
    plt.subplot(121)
    plt.boxplot(maxplots, labels=label)
    plt.title('Box-plot of max-rate for N = {}'.format(args.n))
    plt.grid(linewidth=0.2)
    plt.ylabel('Approximation Rate')
    plt.savefig('Approximation rate box-plot.png')

    plt.subplot(122)
    plt.boxplot(avgplots, labels=label)
    plt.title('Box-plot of avg-rate for N = {}'.format(args.n))
    plt.grid(linewidth=0.2)
    plt.ylabel('Approximation Rate')
    plt.savefig('img/Approximation rate box-plot.png')
    plt.show()

def factor_analysis():
    assert args.factor_analysis == 'True', 'Wrong argument for factor analysis mode. '
    N_list = [10, 50, 100, 500, 1000]
    k_list = [2, 3, 4, 5, 10]

    dict_list = []
    if args.mode == 'tree':
        """
        We analyze varying N 
        """
        txt=''
        var = 'N'
        my_list = N_list
        for n in my_list:
            args.n = n 
            result_dict = evaluate_approx(get_graphs())
            dict_list.append(result_dict)

    elif args.mode == 'spanner':
        """
        We analyze varying k
        """
        args.n = 1000 # Change as you wish
        txt = '(N={})'.format(args.n)
        var = 'k'
        
        my_list = k_list 
        for k in my_list:
            args.k = k
            result_dict = evaluate_approx(get_graphs())
            dict_list.append(result_dict)
    
    labels = []
    max_dict = {key: [] for key in args.dataset_list}
    avg_dict = {key: [] for key in args.dataset_list}
    
    for idx, result_dict in enumerate(dict_list):
        lbl = '{}={}'.format(var, my_list[idx])
        labels.append(lbl)

        for key_g, value_g in result_dict.items():
            _, (max_list, avg_list) = value_g
            max_dict[key_g].append(max_list)
            avg_dict[key_g].append(avg_list)


    for key in args.dataset_list:
        max_means = np.array(max_dict[key]).mean(axis=1).flatten()
        print(max_means)
        max_stds = np.array(max_dict[key]).std(axis=1).flatten()

        avg_means = np.array(avg_dict[key]).mean(axis=1).flatten()
        avg_stds = np.array(max_dict[key]).std(axis=1).flatten()

        ind = np.arange(len(max_means))
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind - width/2, max_means, width, yerr=max_stds,
                        color='SkyBlue', label='Maximum distortion')
        rects2 = ax.bar(ind + width/2, avg_means, width, yerr=avg_stds,
                        color='IndianRed', label='Mean distortion')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Approximation Rate')
        ax.set_title('Maximum and average distortion rates of {} per different {} {}'.format(args.mode, var, txt))
        ax.set_xticks(ind)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects, xpos='center'):
            """
            Attach a text label above each bar in *rects*, displaying its height.

            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.
            """

            xpos = xpos.lower()  # normalize the case of the parameter
            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                        '{}'.format(height), ha=ha[xpos], va='bottom')

        autolabel(rects1, "left")
        autolabel(rects2, "right")

        plt.show()

    return dict_list


def _plot_runtime(result_dict):
    for key_g, value_g in result_dict.items():
        (mean, full) = value_g
        full = np.array(full).transpose((2,0,1))
        for p in full:
            plt.loglog(p[:,0], p[:,1], basex=2, basey=2, c='coral')

        plot = np.array(mean)
        slope, _ = np.polyfit(np.log2(plot[:,0]), np.log2(plot[:,1]), 1)
        plt.loglog(plot[:,0], plot[:,1], basex=2, basey=2, c='steelblue', label=('Mean plot'))
        plt.legend()
        plt.grid(linewidth=0.2)
        plt.title('Slope: {:.3f}'.format(slope))
        plt.xlabel('Number of nodes (N)')
        plt.ylabel('Time (s)')
        plt.savefig('img/{}_runtime_n={}.png'.format(args.mode, args.n))
        plt.show()

def plot(evaluation):
    if args.eval_type == 'approx':
        _plot_approx(evaluation)
    elif args.eval_type == 'runtime':
        _plot_runtime(evaluation)
    else:
        raise Exception('Wrong evaluation type.')


#################################################################################
#						     	      Main										#
#################################################################################
if __name__ == '__main__':
    # Setup
    args = get_args()
    log_dir = 'logs'
    details = 'graphs={}:{}_eval={}_mode={}_n={}_k={}_repeats={}'.format(args.evaluation, 
                                                                         args.dataset_list, 
                                                                         args.eval_type, 
                                                                         args.mode, 
                                                                         args.n, 
                                                                         args.k, 
                                                                         args.repeats) 
    log_name = '{}.log'.format(details)
    set_logger(log_dir, log_name)
    
    logging.info('Start evaluation program [details: {}].'.format(details))

    

    if args.factor_analysis == 'False':
        # Create graphs 
        logging.info('Creating {} graphs {}.'.format(args.evaluation, args.dataset_list))
        input_graphs = get_graphs()
        logging.info('- Done.')
        if args.eval_type == 'approx':
            # Evaluate approximation rate 
            logging.info('Begin approximation rate evaluation.')
            evaluation = evaluate_approx(input_graphs)
            logging.info('- Done.')
        elif args.eval_type == 'runtime':
            # Evaluate runtime  
            logging.info('Begin runtime evaluation.')
            evaluation = evaluate_runtime()
            logging.info('- Done.') 
        else: 
            raise Exception('Wrong evaluation type. ')

        # Visualization# Evaluate approximation rate 
        logging.info('Begin plotting the results.')
        plot(evaluation)
        logging.info('- Done.')

    elif args.factor_analysis == 'True':
        logging.info('Begin factor analysis evaluation.')
        evaluation = factor_analysis()      
        logging.info('- Done.')
    
    logging.info('Saving results.') 
    import pickle as pk
    if not os.path.exists('save'):
        os.makedirs('save')
    pk.dump(evaluation, open( "save/{}_eval.pkl".format(details), "wb" ))
    pk.dump(args, open( "save/{}_args.pkl".format(details), "wb" ))
    logging.info('- Done.')

    logging.info('Evaluation program complete.')


