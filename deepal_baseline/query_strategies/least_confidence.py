import numpy as np
import torch
from .strategy import Strategy

# this is least confidence

class LeastConfidence(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidence, self).__init__(dataset, net)

    def query(self, n):
        # Get the indices and data of unlabeled points
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Predict probabilities for unlabeled data
        probs = self.predict_prob(unlabeled_data)

        # Calculate uncertainty based on the most confident prediction
        uncertainties = 1 - probs.max(1)[0]

        # Return the indices of the most uncertain points
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
