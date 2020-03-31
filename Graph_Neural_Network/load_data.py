import os
import pickle

import deepchem as dc
import numpy as np
import pandas as pd
import torch
from deepchem.data.datasets import NumpyDataset
from deepchem.feat.mol_graphs import ConvMol
from mordred import Calculator, descriptors
from numpy import flip
from rdkit import Chem
from utils import preprocess_adj


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency, dtype=float)


def save_feature(dir, Features, Normed_adj, Interactions, smiles, edge, full_feature, dataset=None):
    dir_input = (dir + dataset + '/')
    os.makedirs(dir_input, exist_ok=True)
    np.save(dir_input + 'Features', Features)
    np.save(dir_input + 'Normed_adj', Normed_adj)
    np.save(dir_input + 'Interactions', Interactions)
    np.save(dir_input + 'smiles', smiles)

    with open(dir_input + 'edge', 'wb') as f:
        pickle.dump(edge, f)

    with open(dir_input + 'full_feature', 'wb') as a:
        pickle.dump(full_feature, a)


def fix_input(feature_array, iAdjTmp):
    "Fix number of input molecular atoms"
    maxNumAtoms = 16
    iFeature = np.zeros((maxNumAtoms, 75))
    if len(feature_array) <= maxNumAtoms:
        iFeature[0:len(feature_array), 0:75] = feature_array
    else:
        iFeature = feature_array[0:maxNumAtoms]

    adjacency = np.zeros((maxNumAtoms, maxNumAtoms))

    if len(feature_array) <= maxNumAtoms:
        adjacency[0:len(feature_array), 0:len(feature_array)] = iAdjTmp
    else:
        adjacency = iAdjTmp[0:maxNumAtoms, 0:maxNumAtoms]

    return iFeature, adjacency


def get_feature(dataset):
    Features_decrease, adj_decrease, edge_decrease, full_feature_decrease = [], [], [], []
    Interactions, smiles = [], []

    for x, label, w, smile in dataset.itersamples():

        # The smile is used to extract molecular fingers
        smiles.append(smile)

        interaction = label
        Interactions.append(np.array([interaction]))

        mol = Chem.MolFromSmiles(str(smile))

        if not mol:
            raise ValueError("Could not parse SMILES string:", smile)

        # increased order
        feature_increase = x.get_atom_features()
        iAdjTmp_increase = create_adjacency(mol)

        # decreased order
        # Turn the data upside down
        feature_decrease = flip(feature_increase, 0)
        iAdjTmp_decrease = flip(iAdjTmp_increase, 0)

        # Obtaining fixed-size molecular input data
        iFeature_decrease, adjacency_decrease = fix_input(feature_decrease, iAdjTmp_decrease)

        Features_decrease.append(np.array(iFeature_decrease))
        normed_adj_decrease = preprocess_adj(adjacency_decrease)
        adj_decrease.append(normed_adj_decrease)

        # Transforms data into PyTorch Geometrics specific data format.
        index = np.array(np.where(iAdjTmp_decrease == 1))
        edge_index = torch.from_numpy(index).long()
        edge_decrease.append(edge_index)

        feature = torch.from_numpy(feature_decrease.copy()).float()
        full_feature_decrease.append(feature)
    return Features_decrease, adj_decrease, edge_decrease, full_feature_decrease, Interactions, smiles


def load_data(dir, filename, fingerprint_size=100, prediction=False):
    # Load GA dataset
    data = pd.read_csv(filename)
    data = data.sample(frac=1)
    smiles = data['SMILES'].to_list()
    conv = []
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    for smile in smiles:
        conv.append(Chem.MolFromSmiles(smile))
    graphs = featurizer.featurize(conv)

    if prediction:
        scores = [0 for _ in range(len(smiles))]
        predict_dataset = NumpyDataset(graphs, scores, ids=smiles)
        Features_decrease1, adj_decrease1, edge_decrease1, full_feature_decrease1, Interactions1, smiles1 = get_feature(
            predict_dataset)
        save_feature(dir, Features_decrease1, adj_decrease1, Interactions1, smiles1, edge_decrease1,
                     full_feature_decrease1,
                     dataset='predict_data')
    else:
        scores = data['score'].to_list()
        step = len(smiles) // 10
        test_dataset = NumpyDataset(graphs[:step], scores[:step], ids=smiles[:step])
        valid_dataset = NumpyDataset(graphs[step:2 * step + 1], scores[step:2 * step + 1],
                                     ids=smiles[step:2 * step + 1])
        train_dataset = NumpyDataset(graphs[2 * step + 1:], scores[2 * step + 1:], ids=smiles[2 * step + 1:])

        # Create files of graph information
        Features_decrease1, adj_decrease1, edge_decrease1, full_feature_decrease1, Interactions1, smiles1 = get_feature(
            train_dataset)
        Features_decrease2, adj_decrease2, edge_decrease2, full_feature_decrease2, Interactions2, smiles2 = get_feature(
            valid_dataset)
        Features_decrease3, adj_decrease3, edge_decrease3, full_feature_decrease3, Interactions3, smiles3 = get_feature(
            test_dataset)
        save_feature(dir, Features_decrease1, adj_decrease1, Interactions1, smiles1, edge_decrease1,
                     full_feature_decrease1,
                     dataset='train_data')
        save_feature(dir, Features_decrease2, adj_decrease1, Interactions2, smiles2, edge_decrease2,
                     full_feature_decrease2,
                     dataset='valid_data')
        save_feature(dir, Features_decrease3, adj_decrease1, Interactions3, smiles3, edge_decrease3,
                     full_feature_decrease3,
                     dataset='test_data')

    # Creat the fingerprints based on mordred
    calc = Calculator(descriptors, ignore_3D=True)
    if prediction:
        datasets = ['predict_data']
    else:
        datasets = ["train_data", "valid_data", "test_data"]

    for inc, dataset in enumerate(datasets):
        data = np.load(dir + dataset + "/smiles.npy")
        alldes = []
        for smiles in data:
            mol = Chem.MolFromSmiles(smiles)
            alldes.append(calc(mol)[:fingerprint_size])
        np.save(dir + dataset + "/fingerprint_stand.npy", np.array(alldes))
