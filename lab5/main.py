import pickle
import pandas as pd
import networkx as nx

from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

from lab4.materials.longae.utils import create_adj_from_edgelist
from lab4.materials.longae.models.ae import autoencoder
from lab4.main import get_mappings, read_graph_largest_component

CORA_CITES_LARGEST_COMPONENT_FILE_PATH = '../data/cora/cora.cites.largest'
ENCODER_EPOCH_FILE_PATH = '../data/cora/graph_encoder_epoch_49.h5'
CORA_CITES_FILE_PATH = '../data/cora/cora.cites'
CORA_CONTENT_FILE_PATH = '../data/cora/cora.content'
GIRVAN_PREDS_DUMP_FILE_PATH = '../data/cora/girvan_preds.dump'


def get_node_data(max_cc_nodes):
    feature_names = ['w_{}'.format(ii) for ii in range(1433)]
    column_names = feature_names + ['subject']
    node_data = pd.read_table(CORA_CONTENT_FILE_PATH,
                              header=None, names=column_names)

    # we remove the data about the nodes that are not in the largest connected component
    nodes_to_drop = [node for node in node_data.index.values.tolist() if node not in max_cc_nodes]
    node_data = node_data.drop(labels=nodes_to_drop)

    return node_data


def save_girvan_newman_predictions(g):
    communities = nx.algorithms.community.girvan_newman(g)

    i = 0
    while i < 6:  # in the first iteration we get 2 communities, the second 3 communities, and so on...
        comm = next(communities)
        print(f"Iteration {i}")
        if i == 5:  # by iteration 6 (i == 5) we have 7 communities
            pickle.dump(comm, open(GIRVAN_PREDS_DUMP_FILE_PATH, 'wb'))
        i = i + 1


def get_class_encoding(classes):
    # key -> subject; value -> int representation of 'key'
    # enumerate(classes) -> list of tuples (i, element), where 'element' is the subject and
    # 'i' will be the int representation of 'element'
    return dict(map(lambda kv: (kv[1], kv[0]), enumerate(classes)))


def get_sorted_labels(node_labels):
    return [label for node, label in sorted(node_labels.items(), key=lambda x: x[0])]


def get_labels_true(data):
    # we convert the classes (subjects) into numerical values from the range [0-6]
    class_encoder = get_class_encoding(data['subject'].unique())
    # map where the key is a node in the graph, and the value is the int representation of it's class
    node_label = dict(map(lambda kv: (kv[0], class_encoder[kv[1]]), data['subject'].to_dict().items()))
    # then we sort the map by the keys (nodes) and return only the values (labels)
    # we sort by nodes because the order of the nodes in the lists we send to the metrics functions matters
    # i.e. the two lists, 'label_true' and 'label_pred' represent the same nodes
    # ex. label_true[0] = 1; label_pred[0] = 1 => means that the true community of node represented by index '0' is '1'
    # and also the predicted one (label_pred[0]) is also '1'; we need to make sure that the the nodes represented
    # by index '0' in both lists, is actually the same node (that's why we sort the pred_labels too)
    return get_sorted_labels(node_label)


def get_labels_pred_girvan_newman(g):
    # the algorithm ('girvan_newman') runs for about ~2h
    # so i ran it once, then saved the returned object to a file
    # save_girvan_newman_predictions(g)  # COMMENT THIS

    # 'predicted_communities' is a tuple of sets; each element in the tuple represents a community
    predicted_communities = pickle.load(open(GIRVAN_PREDS_DUMP_FILE_PATH, 'rb'))
    node_pred_label = {}
    # we use the index of the community in the tuple as a label (in range [0-6])
    for label, pred_comm in enumerate(predicted_communities):
        for node in pred_comm:
            node_pred_label[node] = label
    return get_sorted_labels(node_pred_label)


def get_labels_pred_spectral_clustering(g):
    # 'precomputed' because we have the adj matrix
    model = SpectralClustering(
        n_clusters=7, affinity='precomputed', n_init=100, assign_labels='discretize'
    )
    adj_mat = nx.to_pandas_adjacency(g)
    predicted_communities = model.fit_predict(adj_mat.to_numpy())

    node_pred_label = {}
    node_to_index, index_to_node = get_mappings(adj_mat)
    # 'predicted_communities' is a numpy.array with shape (num_nodes,) where each index represents a node
    # (that's why we need the 'index_to_node' mapping) and each element is the predicted community for the node
    # represented by that index
    for i, pred in enumerate(predicted_communities):
        node_pred_label[index_to_node[i]] = pred
    return get_sorted_labels(node_pred_label)


def get_labels_pred_kmeans(g):
    # we write the largest component to a edge list file, so later we can pass it on to the command:
    # python longae/train_reconstruction.py [CORA_CITES_LARGEST_COMPONENT_FILE_PATH] 1
    nx.write_edgelist(g, path=CORA_CITES_LARGEST_COMPONENT_FILE_PATH)
    # using the functions from the 'longae' module
    # first, use: python longae/train_reconstruction.py [CORA_CITES_LARGEST_COMPONENT_FILE_PATH] 1
    adj = create_adj_from_edgelist(CORA_CITES_LARGEST_COMPONENT_FILE_PATH)
    encoder = autoencoder(sparse_net=False, adj=adj)[0]
    encoder.load_weights(ENCODER_EPOCH_FILE_PATH)
    encoded_adj = encoder.predict([adj])
    predicted_communities = KMeans(n_clusters=7, n_init=100).fit_predict(encoded_adj)

    node_pred_label = {}
    pandas_adj = nx.to_pandas_adjacency(g)
    node_to_index, index_to_node = get_mappings(pandas_adj)
    # 'predicted_communities' is a numpy.array with shape (num_nodes,) where each index represents a node
    # (that's why we need the 'index_to_node' mapping) and each element is the predicted community for the node
    # represented by that index
    for i, pred in enumerate(predicted_communities):
        node_pred_label[index_to_node[i]] = pred
    return get_sorted_labels(node_pred_label)


def calculate_metrics(true, pred):
    print(f'Adjusted mutual info: {adjusted_mutual_info_score(true, pred)}')
    print(f'Normalized mutual info: {normalized_mutual_info_score(true, pred)}\n')


if __name__ == '__main__':
    cora_g = read_graph_largest_component()
    nodes_data = get_node_data(nx.nodes(cora_g))
    labels_true = get_labels_true(nodes_data)

    ###############################################################

    print('GIRVAN NEWMAN')
    labels_pred_girvan_newman = get_labels_pred_girvan_newman(cora_g)
    calculate_metrics(labels_true, labels_pred_girvan_newman)

    ###############################################################

    print('SPECTRAL CLUSTERING')
    labels_pred_spectral = get_labels_pred_spectral_clustering(cora_g)
    calculate_metrics(labels_true, labels_pred_spectral)

    ###############################################################

    print('KMEANS WITH ENCODED NODES')
    labels_pred_kmeans = get_labels_pred_kmeans(cora_g)
    calculate_metrics(labels_true, labels_pred_kmeans)
