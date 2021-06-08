import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset
import numpy as np 
from . import fusion
from sklearn.cluster import SpectralClustering

class ANF_net(nn.Module):
    def __init__(self, input_size, input_channels, output_size):
        super().__init__()

        self.weight = nn.Conv1d(input_channels, 1, 1)

        self.fc = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )

    def forward(self, input_):
        output = self.weight(input_)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

class ANFDataset(Dataset):
    def __init__(self, datapath, cancer_type):
        '''
        datapath : the dir where ANF dataset is in
        cancer_type : str, one of the five cancer types
        true_label : bool, use true label or ANF pred label
        '''

        feature_types = ['fpkm', 'methy450', 'mirnas']
        id2subtype = pd.read_table(datapath + 'project_ids.txt', sep=' ')
        id2subtype.x = pd.factorize(id2subtype.x)[0]
        id2subtype = id2subtype['x']

        # calculate Wall : kNN graph as network input
        affinities = [pd.read_table(datapath + cancer_type + '_' + featuretype + '_.txt', sep=' ') 
            for featuretype in feature_types]
        graphs = [np.array(affinity) for affinity in affinities]
        self.Wall = fusion.kNN_pruned_list(graphs) 

        self.y_true = np.array([id2subtype[id] for id in affinities[0].index])
        self.y_true, uniques = pd.factorize(self.y_true)
        self.y_true = torch.tensor(self.y_true, dtype=torch.long)
        n_cluster = len(np.unique(self.y_true))

        # predict label using ANF
        fuse = fusion.affinity_graph_fusion(graphs)
        fuse = (fuse + fuse.T)/2
        self.y_pred = SpectralClustering(n_clusters=n_cluster, random_state=0, affinity='precomputed').fit_predict(fuse)
        self.y_pred = torch.tensor(self.y_pred, dtype=torch.long)

    def __len__(self):
        return len(self.y_true)

    def __getitem__(self, idx):
        data = torch.tensor([W[idx] for W in self.Wall], dtype=torch.float32)
        label = (self.y_true[idx], self.y_pred[idx])
        return data, label