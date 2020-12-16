import os
import numpy as np
import pandas as pd
import networkx as nx
from gem.embedding.lap import LaplacianEigenmaps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import feature_extraction, model_selection

CORA_CITES_FILE_PATH = '../data/cora/cora.cites'
CORA_CONTENT_FILE_PATH = '../data/cora/cora.content'
LAPLACIAN_EMB_FILE_PATH = '../data/cora/embeddings/laplacian_50.emb'
NODE2VEC_EMB_FILE_PATH = '../data/cora/embeddings/node2vec_50.emb'
SDNE_EMB_FILE_PATH = '../data/cora/embeddings/sdne_50.emb'


def read_graph():
    # Load the graph from edgelist
    edgelist = pd.read_table(CORA_CITES_FILE_PATH,
                             header=None,
                             names=['target',
                                    'source'])  # swapped 'target' & 'source' -> last line in Cora's README

    edgelist['label'] = 'cites'
    graph = nx.from_pandas_edgelist(
        edgelist,
        source='source',
        target='target',
        edge_attr='label',
        create_using=nx.DiGraph  # the graph needs to be directed -> 'create_using=nx.DiGraph'
    )
    nx.set_node_attributes(graph, 'paper', 'label')

    # Load the features and subject for the nodes
    feature_names = ['w_{}'.format(ii) for ii in range(1433)]
    column_names = feature_names + ['subject']
    node_data = pd.read_table(CORA_CONTENT_FILE_PATH,
                              header=None, names=column_names)

    return graph, node_data, feature_names


def split_data(node_data):
    train_data, test_data = model_selection.train_test_split(node_data, train_size=0.7, test_size=None,
                                                             stratify=node_data['subject'])
    return train_data, test_data


def encode_classes(train_data, test_data, is_pandas_data_frame=True):
    target_encoding = feature_extraction.DictVectorizer(sparse=False)

    trd = train_data
    tsd = test_data
    if is_pandas_data_frame:
        trd = train_data[['subject']].to_dict('records')
        tsd = test_data[['subject']].to_dict('records')

    train_targets = target_encoding.fit_transform(trd)
    test_targets = target_encoding.transform(tsd)

    return train_targets, test_targets


def calculate_metrics(test_targets, predictions):
    """Calculation of accuracy score, F1 micro and F1 macro"""
    print(f'\tAccuracy score: {accuracy_score(test_targets, predictions)}')
    print(f'\tF1-micro: {f1_score(test_targets, predictions, average="micro")}')
    print(f'\tF1-macro: {f1_score(test_targets, predictions, average="macro")}')


def add_neighbour_data(g, segment, all):
    features = {}
    for node in segment.index.values.tolist():
        node_features = segment.loc[node][:-1].to_dict()  # getting features without the class
        for neighbour in nx.neighbors(g, node):
            # we search into 'all' because the neighbour may not be in the 'segment'
            neighbour_features = all.loc[neighbour][:-1].to_dict()  # we skip the class i.e. subject ([:-1])
            for k, v in neighbour_features.items():
                # if the current node does not already have the feature 'k' (node_features[k] == 0) and only if the
                # neighbour has that feature (v == 1) too, then we can say that this nodes has that feature
                if node_features[k] == 0 and v == 1:
                    node_features[k] = 1
        features[node] = node_features
    return pd.DataFrame.from_dict(features, orient='index')


def create_and_evaluate_classifier(train_features, train_targets, test_features, test_targets):
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(train_features, train_targets)
    predictions = classifier.predict(test_features)
    calculate_metrics(test_targets, predictions)


def save_embeddings(file_path, embs, nodes):
    """Save node embeddings

    :param file_path: path to the output file
    :type file_path: str
    :param embs: matrix containing the embedding vectors
    :type embs: numpy.array
    :param nodes: list of node names
    :type nodes: list(int)
    :return: None
    """
    with open(file_path, 'w') as f:
        f.write(f'{embs.shape[0]} {embs.shape[1]}\n')
        for node, emb in zip(nodes, embs):
            f.write(f'{node} {" ".join(map(str, emb.tolist()))}\n')


def read_embeddings(file_path):
    """ Load node embeddings
    :param file_path: path to the embedding file
    :type file_path: str
    :return: dictionary containing the node names as keys
    and the embeddings vectors as values
    :rtype: dict(int, numpy.array)
    """
    with open(file_path, 'r') as f:
        f.readline()
        embs = {}
        line = f.readline().strip()
        while line != '':
            parts = line.split()
            embs[int(parts[0])] = np.array(list(map(float, parts[1:])))
            line = f.readline().strip()
    return embs


def extract_data_from_embs(embs, train_nodes, test_nodes):
    train_subjects = []
    test_subjects = []
    train_features = {}
    test_features = {}

    for i, (node, vec) in enumerate(embs.items()):
        if node in train_nodes.index.values.tolist():
            train_features[node] = dict([(i, x) for i, x in enumerate(vec)])  # doing this because of pandas
            train_subjects.append({'subject': train_nodes.loc[node][-1]})  # doing this so that i can use the same 'encode_classes' function
        elif node in test_nodes.index.values.tolist():
            test_features[node] = dict([(i, x) for i, x in enumerate(vec)])
            test_subjects.append({'subject': test_nodes.loc[node][-1]})

    train_targets, test_targets = encode_classes(train_subjects, test_subjects, False)
    train_features, test_features = pd.DataFrame.from_dict(train_features, orient='index'), pd.DataFrame.from_dict(test_features, orient='index')
    return train_features, train_targets, test_features, test_targets


if __name__ == '__main__':
    g, nodes, features_names = read_graph()
    train_data, test_data = split_data(nodes)
    train_targets, test_targets = encode_classes(train_data, test_data)

    ###############################################################

    no_neighbour_train_features, no_neighbour_test_features = train_data[features_names], test_data[features_names]
    neighbour_train_features = add_neighbour_data(g, train_data, nodes)
    neighbour_test_features = add_neighbour_data(g, test_data, nodes)

    print('Score without neighbours data')
    create_and_evaluate_classifier(no_neighbour_train_features, train_targets, no_neighbour_test_features, test_targets)
    print('Score with neighbours data')
    # we can use the same '[]_targets' because both (with neighbours and without) datasets are extracted from the same
    # datasets: 'train_data' and 'test_data' and follow the same order of nodes
    create_and_evaluate_classifier(neighbour_train_features, train_targets, neighbour_test_features, test_targets)

    ###############################################################

    if not os.path.exists(LAPLACIAN_EMB_FILE_PATH):
        laplacian = LaplacianEigenmaps(d=50)
        embs = laplacian.learn_embedding(g, edge_f=None, is_weighted=False, no_python=True)
        save_embeddings(LAPLACIAN_EMB_FILE_PATH, embs[0], list(g.nodes))

    laplacian_embs = read_embeddings(LAPLACIAN_EMB_FILE_PATH)
    node2vec_embs = read_embeddings(NODE2VEC_EMB_FILE_PATH)  # we already have the embeddings
    sdne_embs = read_embeddings(SDNE_EMB_FILE_PATH)  # we already have the embeddings

    print('LAPLACIAN')
    create_and_evaluate_classifier(*extract_data_from_embs(laplacian_embs, train_data, test_data))
    print('NODE2VEC')
    create_and_evaluate_classifier(*extract_data_from_embs(node2vec_embs, train_data, test_data))
    print('SDNE')
    create_and_evaluate_classifier(*extract_data_from_embs(sdne_embs, train_data, test_data))
