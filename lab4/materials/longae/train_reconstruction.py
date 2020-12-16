"""
This script trains a model based on the symmetrical autoencoder
architecture with parameter sharing. The model performs reconstruction
using only latent features learned from local graph topology.

Usage: python train_reconstruction.py <dataset_path> <gpu_id>
"""
import sys
import numpy as np
from tensorflow.python.keras import backend as K

from utils import create_adj_from_edgelist, compute_precisionK
from utils import generate_data
from models.ae import autoencoder

import os

if len(sys.argv) < 3:
    print('\nUSAGE: python %s <dataset_path> <gpu_id>' % sys.argv[0])
    sys.exit()
dataset_path = sys.argv[1]
gpu_id = sys.argv[2]

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

print('\nLoading dataset {:s}...\n'.format(dataset_path))
try:
    adj = create_adj_from_edgelist(dataset_path)
except IOError:
    sys.exit('Bad reading of the graph')

original = adj.copy()
train = adj.copy()

print('\nCompiling autoencoder model...\n')
encoder, ae = autoencoder(sparse_net=False, adj=adj)
print(ae.summary())

# Specify some hyperparameters
epochs = 50

print('\nFitting autoencoder model...\n')

generator = generate_data(original, train)
for e in range(epochs):
    print('\nEpoch {:d}/{:d}'.format(e + 1, epochs))
    print('Learning rate: {:6f}'.format(K.eval(ae.optimizer.lr)))
    curr_iter = 0
    train_loss = []
    batch_adj, batch_train = next(generator)
    # Each iteration/loop is a batch of train_batch_size samples
    res = ae.train_on_batch([batch_adj], [batch_train])
    train_loss.append(res)

    train_loss = np.asarray(train_loss)
    train_loss = np.mean(train_loss, axis=0)
    print('Avg. training loss: {:6f}'.format(train_loss))
    encoder.save_weights(f'data/cora/graph_encoder_epoch_{e}.h5')

print('\nEvaluating reconstruction performance...')
reconstruction = ae.predict([original])
print('Computing precision@k...')
k = [1, 5, 10, 500]

precisionK = compute_precisionK(original, reconstruction, np.max(k))
for index in k:
    if index == 0:
        index += 1
    print('Precision@{:d}: {:6f}'.format(index, precisionK[index - 1]))
print('\nAll Done.')
