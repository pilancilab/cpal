import numpy as np
import torch
import copy
from .strategy import Strategy

class QueryByCommittee(Strategy):
    def __init__(self, dataset, base_net, n_committee=3, random_states=None):
        super(QueryByCommittee, self).__init__(dataset, base_net)
        self.n_committee = n_committee
        self.random_states = random_states or [None] * n_committee

    def query(self, n):
        # Get the unlabeled data
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        
        # Create committee by training multiple models with different random seeds
        committee_probs = []
        for i in range(self.n_committee):
            net_copy = copy.deepcopy(self.net)
            net_copy.initialize()
            # Set the seed for each committee member
            torch.manual_seed(self.random_states[i])
            # Fit the model on labeled data
            X_labeled, y_labeled = self.dataset.get_labeled_data()
            net_copy.fit(X_labeled, y_labeled)
            # Predict probabilities for the unlabeled data
            probs = net_copy.predict_proba(unlabeled_data)
            committee_probs.append(probs)
        
        # Convert to numpy array for easier manipulation
        committee_probs = np.stack(committee_probs)
        
        # Calculate disagreement between committee members using vote entropy
        avg_probs = np.mean(committee_probs, axis=0)
        log_probs = np.log(avg_probs + 1e-7)
        vote_entropy = -np.sum(avg_probs * log_probs, axis=1)
        
        # Return the indices of the most uncertain samples
        return unlabeled_idxs[vote_entropy.argsort()[-n:]]
