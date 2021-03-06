import numpy as np

def kNN_graph(graph, K):
    '''returns kNN graph of the input graph, for each node only the
    largest k edges are preserved.

    Parameters
    ----------
    graph : 2d numpy array
        The original graph to calculate kNN
    K : int
        number of k nearest neighbors
    '''

    nrow = graph.shape[0]
    idx = np.argsort(graph, axis=1)
    res = np.copy(graph)
    for i in range(nrow):
        res[i, idx[i, :nrow - K - 1]] = 0

    res = res / np.sum(res, axis=1).reshape(-1, 1)
    return res


def affinity_graph_fusion(graphs, K=20, r = 2, weight = None):
    '''Return fused affinity graph using ANF.

    Parameters
    ----------
    graphs : list
        list of affinity graphs in different view
    K : int
        number of K nearest neighbors
    r : int
        steps of random walk, 1 or 2
    weight : list of int
        weight of each view. If None, uniform weight is applied
    '''

    n_graph = len(graphs)
    n_patients = graphs[0].shape[0]

    if weight == None:
        weight = np.ones(n_graph)
    weight /= np.sum(weight)
    
    # row normalization : graph -> transition
    transitions = [graph / np.sum(graph, axis = 1).reshape(-1, 1)
        for graph in graphs]

    # kNN prune : transition -> S
    Sall = [kNN_graph(transition, K) for transition in transitions]

    # fusion : S -> W
    Wall = [0 for i in range(n_graph)]
    if r == 1:
        Wall = graphs
    if r == 2:
        for i in range(n_graph):
            # Sbar : complementary view of S
            Sbar  = np.zeros((n_patients, n_patients))
            sumWeight = sum(weight) - weight[i]
            for k in range(n_graph):
                if k != i:
                    Sbar = Sbar + weight[k]/sumWeight * Sall[k]
            
            # random walk
            Wall[i] = (Sall[i] @ Sbar + Sbar @ Sall[i]) / 2
    
    W = np.zeros((n_patients, n_patients))
    for i in range(n_graph):
        W += weight[i] * Wall[i]
    
    return W

