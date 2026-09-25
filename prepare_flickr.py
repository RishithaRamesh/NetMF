#!/usr/bin/env python

import numpy as np
import scipy.io
import scipy.sparse as sparse

NUM_NODES = 80513
NUM_GROUPS = 195

EDGES_FILE = "data/Flickr-dataset/data/edges.csv"
GROUP_EDGES_FILE = "data/Flickr-dataset/data/group-edges.csv"
OUTPUT_FILE = "data/flickr.mat"


def load_network():
    edges = np.loadtxt(EDGES_FILE, delimiter=",", dtype=int)

    source = edges[:, 0] - 1
    target = edges[:, 1] - 1

    rows = np.concatenate([source, target])
    cols = np.concatenate([target, source])
    values = np.ones(len(rows), dtype=np.float64)

    network = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(NUM_NODES, NUM_NODES)
    )

    network.data[:] = 1.0
    network.eliminate_zeros()

    return network


def load_labels():
    memberships = np.loadtxt(
        GROUP_EDGES_FILE,
        delimiter=",",
        dtype=int
    )

    nodes = memberships[:, 0] - 1
    groups = memberships[:, 1] - 1
    values = np.ones(len(nodes), dtype=np.float64)

    labels = sparse.csr_matrix(
        (values, (nodes, groups)),
        shape=(NUM_NODES, NUM_GROUPS)
    )

    labels.data[:] = 1.0

    return labels


def main():
    print("Building adjacency matrix...")
    network = load_network()

    print("Building label matrix...")
    group = load_labels()

    print("Network shape:", network.shape)
    print("Network nonzeros:", network.nnz)
    print("Group shape:", group.shape)
    print("Group memberships:", group.nnz)

    print("Saving:", OUTPUT_FILE)
    scipy.io.savemat(
        OUTPUT_FILE,
        {
            "network": network,
            "group": group
        }
    )

    print("Done.")


if __name__ == "__main__":
    main()