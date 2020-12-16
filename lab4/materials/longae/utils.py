import random
import numpy as np
import networkx as nx
from scipy.io import loadmat
import scipy.sparse as sp
from itertools import combinations

random.seed(1982)


def generate_data(adj, adj_train):
    zipped = zip(adj, adj_train)
    result = []
    for data in zipped:
        result.append(data)
    while True:  # this flag yields an infinite generator
        a = np.vstack([x[0] for x in result])  # .astype(np.float32)
        t = np.vstack([x[1] for x in result])  # .astype(np.float32)
        yield a, t


def lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5):
    from keras import backend as K
    lrate = base_lr * (1.0 - (curr_iter / float(max_iter))) ** power
    K.set_value(model.optimizer.lr, lrate)
    return K.eval(model.optimizer.lr)


def load_mat_data(dataset_str):
    """ dataset_str: protein, metabolic, conflict, powergrid """
    dataset_path = 'data/' + dataset_str + '.mat'
    mat = loadmat(dataset_path)
    if dataset_str == 'powergrid':
        adj = sp.lil_matrix(mat['G'], dtype=np.float32)
        feats = None
        return adj, feats
    adj = sp.lil_matrix(mat['D'], dtype=np.float32)
    feats = sp.lil_matrix(mat['F'].T, dtype=np.float32)
    # Return matrices in scipy sparse linked list format
    return adj, feats


def split_train_test(adj, ratio=0.0):
    upper_inds = [ind for ind in combinations(range(adj.shape[0]), r=2)]
    np.random.shuffle(upper_inds)
    split = int(ratio * len(upper_inds))
    return np.asarray(upper_inds[:split])


def compute_masked_accuracy(y_true, y_pred, mask):
    correct_preds = np.equal(np.argmax(y_true, 1), np.argmax(y_pred, 1))
    num_examples = float(np.sum(mask))
    correct_preds *= mask
    return np.sum(correct_preds) / num_examples


def compute_precisionK(adj, reconstruction, K):
    """ Modified from https://github.com/suanrong/SDNE """
    N = adj.shape[0]
    reconstruction = reconstruction.reshape(-1)
    sortedInd = np.argsort(reconstruction)[::-1]
    curr = 0
    count = 0
    precisionK = []
    for ind in sortedInd:
        x = int(ind) // N
        y = int(ind) % N
        count += 1
        if (adj[x, y] == 1 or x == y):
            curr += 1
        precisionK.append(1.0 * curr / count)
        if count >= K:
            break
    return precisionK


def create_adj_from_edgelist(dataset_path):
    """ dataset_str: arxiv-grqc, blogcatalog """
    with open(dataset_path, 'r') as f:
        header = next(f)
        edgelist = []
        for line in f:
            if line.startswith('#'):
                continue
            i, j = map(int, line.split())
            edgelist.append((i, j))
    g = nx.Graph(edgelist)
    return nx.adjacency_matrix(g).toarray()
