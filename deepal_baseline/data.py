import numpy as np
import torch
from torchvision import datasets
from sklearn.datasets import make_blobs
import math
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 0

class Data:
    def __init__(self, X_all, Y_all, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_all = X_all
        self.Y_all = Y_all
        
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        selected_indices = tmp_idxs[:num]
        self.labeled_idxs[selected_indices] = True
        ### NEW ###
        return selected_indices
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)

    def get_all_data(self):
        return self.handler(self.X_all, self.Y_all)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test
    
    def cal_train_acc(self, preds):
        return 1.0 * (self.Y_train==preds).sum().item() / self.n_pool

    ### NEW ###
    def cal_rmse_test(self, preds):
        # print(len(self.Y_test),len(preds))
        return np.sqrt(mean_squared_error(self.Y_test,preds))
    def cal_rmse_train(self, preds):
        return np.sqrt(mean_squared_error(self.Y_train,preds))
    def cal_rmse_all(self, preds):
        return np.sqrt(mean_squared_error(self.Y_all,preds))

    def cal_r2_test(self, preds):
        return r2_score(self.Y_test,preds)
    def cal_r2_train(self, preds):
        return r2_score(self.Y_train,preds)
    def cal_r2_all(self, preds):
        return r2_score(self.Y_all,preds)

    
### NEW ###

def spiral_xy(i, total_points, spiral_num, n_shape = 50):
    """
    Create the data for a normalized spiral.

    Arguments:
        i runs from 0 to total_points-1.
        total_points is the total number of points in the spiral.
        spiral_num is 1 or -1.
        n_shape is an int which determines the shape of spiral wrt function, unscaled_spiral_xy.
    """
    # Normalize i to always fit in the range [0, 96], which is the original range.
    i_normalized = i * n_shape / (total_points - 1)
    φ = i_normalized / 16 * math.pi
    r = 6.5 * ((104 - i_normalized) / 104)
    x = (r * math.cos(φ) * spiral_num) / 13 + 0.5
    y = (r * math.sin(φ) * spiral_num) / 13 + 0.5  # spiral_num mirrors the orientation of the spiral
    return (x, y)


def spiral(spiral_num, n=100, n_shape = 50):
    """
    Generate either a clockwise (+1) spiral or a counter clockwise (-1) spiral for a total of n points.

    Arguments:
        spiral_num: 1 or -1 determines the orientation of the spiral.
        n: total number of points in a spiral.
    """
    return [spiral_xy(i, n, spiral_num, n_shape) for i in range(n//2)]


def generate_spiral_data(n=100, n_train = 80, n_shape = 50, seed = RANDOM_STATE, default_label = False):
    """
    Generate binary spiral classification data.

    Arguments:
        n: total number of points in a spiral.
        n_train: numbr of training points.
        seed: permutation randomization seed.
        default_label: True uses label +/-1 and False uses label 0/1
    """
    a = spiral(1,n,n_shape)
    b = spiral(-1,n,n_shape)
    # Combine spiral from both orientation as one and scaling feature space from (0,1)^2 to (-1,1)^2
    X_all=2*np.concatenate((a,b),axis=0)-1 
    X_all=np.append(X_all,np.ones((n,1)),axis=1) # Adding bias to the feature space
    # concatenate the labels of the spiral, which is their orientation (+1/-1)
    if default_label:
        y_all=np.concatenate((np.ones(n//2),-np.ones(n//2))) 
    else:
        y_all=np.concatenate((np.ones(n//2),np.zeros(n//2))) # reset -1 to 0.
    # randomize data indices
    np.random.seed(seed)
    idx = np.random.permutation(n)
    X_all = X_all[idx]
    y_all = y_all[idx]
    # Split into training and testing sets
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]
    
    return X_all, y_all, X_train, y_train, X_test, y_test

# def generate_spiral_data(n=100, train_budget=80):
#     d = 3  # Dimensions of the data (x, y, bias)
    
#     def spiral_xy(i, spiral_num):
#         """
#         Create the data for a spiral.
        
#         Arguments:
#             i runs from 0 to 96
#             spiral_num is 1 or -1
#         """
#         φ = i / 16 * math.pi
#         r = 6.5 * ((104 - i) / 104)
#         x = (r * math.cos(φ) * spiral_num) / 13 + 0.5
#         y = (r * math.sin(φ) * spiral_num) / 13 + 0.5
#         return (x, y)
    
#     def spiral(spiral_num):
#         return [spiral_xy(i, spiral_num) for i in range(n // 2)]
    
#     a = spiral(1)
#     b = spiral(-1)
#     X = 2 * np.concatenate((a, b), axis=0) - 1
#     X = np.append(X, np.ones((n, 1)), axis=1)  # Add the bias term
#     y = np.concatenate((np.ones(n // 2), np.zeros(n // 2)))  # Use 0 and 1 for labels
    
#     # Shuffle the data
#     np.random.seed(RANDOM_STATE)
#     idx = np.random.permutation(n)
#     X = X[idx]
#     y = y[idx]
    
#     # Split into training and testing sets
#     X_train, y_train = X[:train_budget], y[:train_budget]
#     X_test, y_test = X[train_budget:], y[train_budget:]
#     #X_test, y_test = X_train, y_train
    
#     return X, y, X_train, y_train, X_test, y_test

### NEW ###
    
def get_Spiral(handler):
    X_all, Y_all, X_train, y_train, X_test, y_test = generate_spiral_data()
    # Convert to tensors
    X_all = torch.tensor(X_all, dtype=torch.float32)
    Y_all = torch.tensor(Y_all, dtype=torch.long)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return Data(X_all, Y_all, X_train, y_train, X_test, y_test, handler)

### NEW ###

# scikit-activeml blob data
def generate_blob_data(n=100, train_budget=80):
    d = 3  # Dimensions of the data (x, y, z)
    
    # Generate 3-dimensional blob data
    X, y = make_blobs(n_samples=n, centers=2, n_features=d, random_state=0)
    
    # Modify y for binary classification to be 1 and -1
    y = y % 2
    y[y == 0] = -1
    
    # Shuffle the data
    np.random.seed(RANDOM_STATE)
    idx = np.random.permutation(n)
    X = X[idx]
    y = y[idx]
    
    # Split into training and testing sets
    X_train, y_train = X[:train_budget], y[:train_budget]
    X_test, y_test = X[train_budget:], y[train_budget:]
    
    return X, y, X_train, y_train, X_test, y_test


def get_Blob(handler):
    X_all, Y_all, X_train, y_train, X_test, y_test = generate_blob_data()
    # Convert to tensors
    X_all = torch.tensor(X_all, dtype=torch.float32)
    Y_all = torch.tensor(Y_all, dtype=torch.long)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return Data(X_all, Y_all, X_train, y_train, X_test, y_test, handler)

### 


### NEW Scikit-activeml regression data ###

def generate_quadratic_regression(n = 100, train_budget = 80, noise = False, noise_param = 0.05):
    x_vals = np.linspace(-1, 1, n)  # Generate n points uniformly spaced between -1 and 1
    y_true = x_vals ** 2  # Calculate y = x^2
    
    # Add some noise to make it a more realistic regression problem
    if noise:
        noise = noise_param * np.random.randn(n)
        y_vals = y_vals + noise
    else:
        y_vals = y_true

    # Adding bias to the data
    X_all = np.column_stack((x_vals, np.ones(n)))  # Shape: (n, 2) where the second column is the bias term
    y_all = y_vals # y_all is unchanged

    # randomly select X,y
    np.random.seed(RANDOM_STATE)
    
    # Randomly select 80 indices
    selected_indices = np.random.choice(np.arange(n), size=train_budget, replace=False)
    unselected_indices = np.setdiff1d(np.arange(n), selected_indices)
    
    # Select the corresponding rows from X and values from Y for train dataset
    X = X_all[selected_indices]
    y = y_all[selected_indices]
    
    # Test set
    X_test = X_all[unselected_indices]
    y_test = y_all[unselected_indices]

    return X_all, y_all, X, y, X_test, y_test


def get_Quadratic(handler):
    X_all, Y_all, X_train, Y_train, X_test, Y_test = generate_quadratic_regression()
    # Convert to tensors
    X_all = torch.tensor(X_all, dtype=torch.float32)
    Y_all = torch.tensor(Y_all, dtype=torch.float32)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    return Data(X_all, Y_all, X_train, Y_train, X_test, Y_test, handler)


### Benchmark Dataset ###

def get_MNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_FashionMNIST(handler):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_SVHN(handler):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data[:40000], torch.from_numpy(data_train.labels)[:40000], data_test.data[:40000], torch.from_numpy(data_test.labels)[:40000], handler)

def get_CIFAR10(handler):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)
