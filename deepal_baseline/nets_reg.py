import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class RegNet:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device
    
    # for regression
    def train(self, data): # passes labeled_data
        n_epoch = self.params['n_epoch']
        self.clf = self.net().to(self.device)
        self.clf.train()

        # Use MSELoss for regression
        criterion = nn.MSELoss()  # Change loss function to MSELoss for regression
        optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        # optimizer = optim.Adam(self.clf.parameters(), lr=1e-3)

        loader = DataLoader(data, shuffle=True, **self.params['train_args'])

        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                # Forward pass
                out, e1 = self.clf(x)
                
                # Ensure that 'out' and 'y' are compatible in terms of shape
                if out.shape != y.shape:
                    out = out.view_as(y)  # Reshape out to match y if necessary

                # Compute loss using MSELoss
                loss = criterion(out, y)
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
    
    # for regression
    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=torch.float32)  # Initialize preds as float for regression
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                preds[idxs] = out.cpu().squeeze()  # Use the raw output for regression
        return preds
    
    
    # For classification (discretized for regression)
    def predict_prob(self, data):
        self.clf.eval()
        n_classes = len(torch.unique(data.Y))
        probs = torch.zeros([len(data), n_classes])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs
    
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs
    
    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings
    
     # Initialize or reinitialize the network and optimizer
    def initialize(self):
        self.clf = self.net().to(self.device)
        self.clf.apply(self._initialize_weights)
        self.optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)  # Example initialization
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        
class Quadratic_Net(nn.Module):
    def __init__(self):
        #super(Quadratic_Net, self).__init__()
        super().__init__() 
        self.fc1 = nn.Linear(2, 160)  # Input is 2-dimensional (x, y, bias)
        self.fc3 = nn.Linear(160, 1)  # Output is continuous

    def forward(self, x):
        x = F.relu(self.fc1(x))
        e1 = x
        x = F.dropout(x, training=self.training) # default 0.5 drop out probability
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 160 # EDIT NEURONS FOR 80/100 of the train (dmat)

