import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN, DirectedGraphSAGE
from stellargraph.mapper import DirectedGraphSAGENodeGenerator

from tensorflow.keras import layers, losses, Model
from sklearn import preprocessing, model_selection
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from lab2.main import calculate_metrics


def encode_classes(train_subjects, test_subjects):
    encoding = preprocessing.LabelBinarizer()  # we have categorical classes
    train_encoding = encoding.fit_transform(train_subjects)
    test_encoding = encoding.transform(test_subjects)
    return train_encoding, test_encoding, encoding


def split_data(data):
    return model_selection.train_test_split(data, train_size=0.7, test_size=None, stratify=data)


def visi(in_layer, out_layer, gen_nodes, nodes, img_path, title, is_sage=False):
    embedding_model = Model(inputs=in_layer, outputs=out_layer)
    embs = embedding_model.predict(gen_nodes)
    visi_data = embs
    if not is_sage:
        visi_data = embs.squeeze(0)
    tsne = TSNE(n_components=2)
    visi_data = tsne.fit_transform(visi_data)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(visi_data[:, 0], visi_data[:, 1], c=nodes.astype("category").cat.codes, cmap="jet", alpha=0.7)
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$", title=title)
    plt.savefig(img_path)


if __name__ == "__main__":
    cora_dataset = sg.datasets.Cora()  # the 'cora' dataset is built-in into stellargraph datasets module
    # it returns a Stellargraph object, and the node subjects (classes)
    # The features (word occurencess) are already built-in into the "stellar_g" object of type Stellargraph
    stellar_g, node_classes = cora_dataset.load(directed=True)
    train_dataset, test_dataset = split_data(node_classes)
    train_targets, test_targets, target_encoding = encode_classes(train_dataset, test_dataset)

    ###############################################################

    # creating GCN model
    gcn_generator = FullBatchNodeGenerator(stellar_g, method="gcn", sparse=False)
    train_gcn_gen = gcn_generator.flow(train_dataset.index, train_targets)
    gcn = GCN(layer_sizes=[16, 16], activations=['relu', 'relu'], generator=gcn_generator, dropout=0.5)  # 2 GCN layers
    gcn_inp, gcn_out = gcn.in_out_tensors()  # for the KERAS model

    # creating KERAS model with the GCN model layers
    gcn_dense_layer = layers.Dense(units=train_targets.shape[1], activation="softmax")(gcn_out)
    keras_gcn = Model(inputs=gcn_inp, outputs=gcn_dense_layer)  # 2 GCN, 1 Dense
    keras_gcn.compile(
        optimizer="adam",
        loss=losses.categorical_crossentropy,
        metrics=["accuracy"],
    )
    keras_gcn.fit(train_gcn_gen, epochs=10, batch_size=32, verbose=1, shuffle=False)  # training the model

    # testing the GCN-KERAS model
    test_gcn_gen = gcn_generator.flow(test_dataset.index)
    gcn_predictions = keras_gcn.predict(test_gcn_gen)
    gcn_predicted_classes = target_encoding.inverse_transform(gcn_predictions.squeeze())
    print("GCN MODEL: 2 GCN layers, 1 Dense layer")
    calculate_metrics(test_dataset.values, gcn_predicted_classes)

    # visualization of the embeddings
    visi(
        in_layer=gcn_inp,
        out_layer=gcn_out,
        gen_nodes=gcn_generator.flow(node_classes.index),
        nodes=node_classes,
        img_path="../data/cora/img/gcn.png",
        title="GCN embs",
        is_sage=False
    )

    ###############################################################

    # creating SAGE model
    batch_size = 50
    num_samples = [10, 5]
    sage_generator = DirectedGraphSAGENodeGenerator(stellar_g, batch_size, num_samples, num_samples)
    train_sage_gen = sage_generator.flow(train_dataset.index, train_targets, shuffle=False)
    sage = DirectedGraphSAGE(layer_sizes=[32, 32], generator=sage_generator, bias=False, dropout=0.5)
    sage_inp, sage_out = sage.in_out_tensors()

    # creating KERAS model with the SAGE model layers
    sage_dense_layer = layers.Dense(units=train_targets.shape[1], activation="softmax")(sage_out)
    keras_sage = Model(inputs=sage_inp, outputs=sage_dense_layer)
    keras_sage.compile(
        optimizer="adam",
        loss=losses.categorical_crossentropy,
        metrics=["acc"]
    )
    keras_sage.fit(train_sage_gen, epochs=10, batch_size=32, verbose=1, shuffle=False)

    # testing the SAGE-KERAS model
    test_sage_gen = sage_generator.flow(test_dataset.index)
    sage_predictions = keras_sage.predict(test_sage_gen)
    sage_predicted_classes = target_encoding.inverse_transform(sage_predictions.squeeze())
    print("SAGE MODEL: 2 SAGE layers, 1 Dense layer")
    calculate_metrics(test_dataset.values, sage_predicted_classes)

    visi(
        in_layer=sage_inp,
        out_layer=sage_out,
        gen_nodes=sage_generator.flow(node_classes.index),
        nodes=node_classes,
        img_path="../data/cora/img/sage.png",
        title="SAGE embs",
        is_sage=True
    )
