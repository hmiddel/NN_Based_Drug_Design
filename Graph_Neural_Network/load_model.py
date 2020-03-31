import torch
import pandas as pd
from training_plot import plots
from Model_train import C_SGEN, T, mydataset
from load_data import load_data, fix_input, create_adjacency
from torch.utils.data import DataLoader
from mordred import Calculator, descriptors
import deepchem as dc
from rdkit import Chem
from numpy import flip
import numpy as np
from utils import preprocess_adj
from torch import Tensor


model = torch.load('data/model/optimized_model')
model.eval()
dir = './data/predictions/'
param = pd.read_csv('data/model/parameters.csv')
batch = param['batch'].to_list()[0]
k = param['k'].to_list()[0]
lr = param['lr'].to_list()[0]
C_SGEN_layers = param['C_SGEN_layers'].to_list()[0]
fingerprint_size = param['fingerprint_size'].to_list()[0]
std = param['std'].to_list()[0]
mean = param['mean'].to_list()[0]
calc = Calculator(descriptors, ignore_3D=True)
device = torch.device('cuda')

def tensoring(list):
    return([torch.FloatTensor(data).to(torch.device('cuda')) for data in list])


class mypredictiondataset(torch.utils.data.Dataset):

    def __init__(self, smiles):
        featurizer = dc.feat.graph_features.ConvMolFeaturizer()
        self.Full_features, self.Full_normed_adj, self.Full_fringer, self.Full_interactions = [], [], [], []
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(str(smile))
            if not mol:
                raise ValueError("Could not parse SMILES string:", smile)

            x = featurizer.featurize([mol])[0]

            # increased order
            feature_increase = x.get_atom_features()
            iAdjTmp_increase = create_adjacency(mol)

            # decreased order
            # Turn the data upside down
            feature_decrease = flip(feature_increase, 0)
            iAdjTmp_decrease = flip(iAdjTmp_increase, 0)

            # Obtaining fixed-size molecular input data
            iFeature_decrease, adjacency_decrease = fix_input(feature_decrease, iAdjTmp_decrease)

            Features_decrease = np.array(iFeature_decrease)
            adj_decrease = preprocess_adj(adjacency_decrease)
            fingerprints = calc(mol)[:fingerprint_size]

            self.Full_features.append(Features_decrease)
            self.Full_normed_adj.append(adj_decrease)
            self.Full_fringer.append(fingerprints)
            self.Full_interactions.append([0])

        self.Full_features = tensoring(self.Full_features)
        self.Full_normed_adj = tensoring(self.Full_normed_adj)
        self.Full_fringer = tensoring(self.Full_fringer)
        self.Full_interactions = tensoring(self.Full_interactions)

        self.dataset = list(zip(np.array(self.Full_features), np.array(self.Full_normed_adj), np.array(self.Full_fringer), np.array(self.Full_interactions)))

    def __getitem__(self, item):
        data_batch = self.dataset[item]

        return data_batch

    def __len__(self):
        return len(self.Full_interactions)


def predict(smile):
    odd = (len(smile)%2 == 1)
    if odd:
        smile.append('c1ccccc1')
    smiles_data = mypredictiondataset(smile)
    predict_loader = DataLoader(smiles_data, batch_size=2)
    tester = T(model.eval(), std, mean, C_SGEN_layers)
    _, _, predicted, _ = tester.test(predict_loader)
    return predicted[:-1] if odd else predicted


if __name__=='__main__':
    data = pd.read_csv('data/qm7.csv')
    data = data.sample(frac=0.1)
    smiles = data['SMILES'].to_list()
    preds = []
    preds += predict(smiles)
    plots(data['score'], preds, label='test pred single smiles', save=True)

    """
    filename = 'data/angel_scores.csv'
    data = pd.read_csv(filename)
    tester = T(model.eval(), std, mean, C_SGEN_layers)
    load_data(dir, filename, fingerprint_size, prediction=True)
    predict_dataset = mydataset('predict_data', dir)
    predict_loader = DataLoader(predict_dataset, batch_size=batch, shuffle=True, drop_last=True)
    _, _, predicted, _ = tester.test(predict_loader)
    plots(data['score'], predicted, label='load_model_test', save=True)
    """