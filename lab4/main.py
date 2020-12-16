import numpy as np
import networkx as nx
from secrets import choice
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import preprocessing

from lab4.materials.random_walk import random_walk

CORA_CITES_FILE_PATH = '../data/cora/cora.cites'


def read_graph_largest_component():
    """
        we'll create an undirected graph from the 'core.cites' document, even tho the graph is directed in nature
        because all the link prediction algorithms in networkx only take an undirected graph as an argument
        (ex. 'nx.adamic_adar_index(nx.Graph, ...)')
    :return: nx.Graph
    """
    g = nx.read_edgelist(CORA_CITES_FILE_PATH, create_using=nx.Graph, nodetype=int)
    # returning the largest connected component
    return nx.subgraph(g, max(nx.connected_components(g), key=len)).copy()  # .copy() so we can modify the graph


def get_negative_random_edges(g, num_random_edges):
    """
    :param g: nx.Graph
    :param num_random_edges: int
    :return: list
    """
    original_edges = list(set(nx.non_edges(g)))  # we'll select from the non-existent (nx.non_edges(...)) edges
    random_edges = []
    while len(random_edges) != num_random_edges:  # every iteration we pick one random edge
        random_edge = choice(original_edges)
        random_edges.append(random_edge)
        original_edges.remove(random_edge)  # we remove it to avoid choosing the same edge multiple times
    return random_edges  # list of tuples (u, v)


def get_and_remove_positive_random_edges(g, num_random_edges):
    """
    :param g: nx.Graph
    :param num_random_edges: int
    :return: list
    """
    original_edges = list(set(nx.edges(g)))  # we'll select from the existent (nx.edges(...)) edges
    random_edges = []
    while len(random_edges) != num_random_edges:  # every iteration TRY to pick one random edge
        random_edge = choice(original_edges)
        g.remove_edge(*random_edge)  # we remove the random edge from the graph
        if not nx.is_connected(g):  # after removing, we need to make sure that the graph is still connected
            g.add_edge(*random_edge)  # if not, we add it back in and try again
        else:  # if the graph remains connected after the removal of 'random_edge', we can keep it as a positive edge
            random_edges.append(random_edge)
            original_edges.remove(random_edge)  # we remove it to avoid choosing the same edge multiple times
    return random_edges  # list of tuples (u, v)


def normalize_sim_values(data):
    """
        Normalizing helps us with choosing the threshold
    :param data: list
    :return: list
    """
    values = np.array([[p for u, v, p in data]])
    normalized_values = preprocessing.normalize(values)
    return [(u, v, normalized_values[0][i]) for i, (u, v, p) in enumerate(data)]


def filter_predictions(predictions, threshold):
    """
    :param predictions: list
    :param threshold: float
    :return: dict
    """
    # key -> (u, v); value -> p
    return dict([((u, v), p) for u, v, p in predictions if p > threshold])


def calculate_metrics(positive, negative, predictions):
    """
        y_true, y_target -> binary value arrays indicating the existence of an edge;
        every index in the arrays represents an edge

        y_true holds the values we want or don't want to see in the predictions
        in our case, we want to see the positive edges, so every positive edge will be represented by '1' in 'y_true'.In
        contrast, we DON'T want to see the negative edges, so every negative edge will be represented by '0' in 'y_true'

        similar to y_true, y_score hold the actual values of the predictions, i.e. the possibility
    :param positive: list
    :param negative: list
    :param predictions: dict
    :return: None
    """
    y_true, y_score = [], []
    for pos_edge, neg_edge in zip(positive, negative):
        y_true.append(1)
        if pos_edge in predictions.keys():
            y_score.append(predictions[pos_edge])
        else:
            y_score.append(0)
        y_true.append(0)
        if neg_edge in predictions.keys():
            y_score.append(predictions[neg_edge])
        else:
            y_score.append(0)
    print(roc_auc_score(y_true, y_score))
    print(average_precision_score(y_true, y_score))


def get_test_nodes(test_edges):
    """
    :param test_edges: list
    :return: set
    """
    nodes = set()  # i'm using a set because some nodes may show up multiple times in 'test_edges'
    for edge in test_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
    return nodes


def get_mappings(adj):
    """
        Returns node labels to adj matrix index mapping, and vise-versa
    :param adj: pandas.Dataframe
    :return: dict, dict
    """
    node_to_index_mapping, index_to_node_mapping = {}, {}
    for i, (node, edges) in enumerate(adj.iterrows()):
        node_to_index_mapping[node] = i
        index_to_node_mapping[i] = node
    return node_to_index_mapping, index_to_node_mapping


if __name__ == '__main__':
    cora_g = read_graph_largest_component()

    cora_original_num_edges = nx.number_of_edges(cora_g)
    num_test_edges = int(cora_original_num_edges * 0.1)

    negative_edges = get_negative_random_edges(cora_g, num_test_edges)
    positive_edges = get_and_remove_positive_random_edges(cora_g, num_test_edges)

    ###############################################################

    # first we need the predicted edges and their similarity values from the different algorithms
    # we pass 'positive + negative' because we want need to see if it will predict the 'negative' ones as well,
    # i.e. the ones we don't want it to predict
    adamic_predictions = set(nx.adamic_adar_index(cora_g, positive_edges + negative_edges))  # tuples (u,v,p)
    jaccard_predictions = set(nx.jaccard_coefficient(cora_g, positive_edges + negative_edges))  # tuples (u,v,p)
    preferential_prediction = set(nx.preferential_attachment(cora_g, positive_edges + negative_edges))  # tuples (u,v,p)

    adamic_predictions = normalize_sim_values(adamic_predictions)
    preferential_prediction = normalize_sim_values(preferential_prediction)
    # jaccard coefficient is already normalized

    adamic_predictions = filter_predictions(adamic_predictions, threshold=0.1)  # tuples (u,v)
    jaccard_predictions = filter_predictions(jaccard_predictions, threshold=0.1)  # tuples (u,v)
    preferential_prediction = filter_predictions(preferential_prediction, threshold=0.1)  # tuples (u,v)

    print('ADAMIC')
    calculate_metrics(positive_edges, negative_edges, adamic_predictions)
    print('JACCARD')
    calculate_metrics(positive_edges, negative_edges, jaccard_predictions)
    print('PREFERENTIAL')
    calculate_metrics(positive_edges, negative_edges, preferential_prediction)

    ###############################################################

    test_nodes = get_test_nodes(positive_edges + negative_edges)
    random_walk_adj_matrix = nx.to_pandas_adjacency(cora_g)
    node_to_index, index_to_node = get_mappings(random_walk_adj_matrix)
    random_walk_adj_matrix = random_walk_adj_matrix.to_numpy()
    random_walk_predictions = []
    for test_node in test_nodes:
        node_predictions = random_walk(random_walk_adj_matrix, node_to_index[test_node])  # numpy.array
        node_predictions = [
            (test_node, index_to_node[i], node_predictions[i]) for i in range(node_predictions.shape[0])
        ]
        random_walk_predictions.extend(node_predictions)  # tuples (u,v,p)

    random_walk_predictions = filter_predictions(random_walk_predictions, threshold=0)  # tuples (u,v)

    print('RANDOM WALK')
    calculate_metrics(positive_edges, negative_edges, random_walk_predictions)

    ###############################################################

    # it writes the graph (without the positive edges), later we execute the script:
    # python longue/train_reconstruction.py data/cora/cora.cites.no.positive 1
    nx.write_edgelist(cora_g, '../data/cora/cora.cites.train', data=False)
