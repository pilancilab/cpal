import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
# Fix locale encoding issue
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="Spiral", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "Spiral"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="EntropySampling", 
                    choices=[#"RandomSampling", 
                             #"LeastConfidence", 
                             #"MarginSampling", 
                             "EntropySampling", 
                             #"LeastConfidenceDropout", 
                             #"MarginSamplingDropout", 
                             #"EntropySamplingDropout", 
                             "KMeansSampling",
                             #"KCenterGreedy", 
                             "BALDDropout", 
                             #"AdversarialBIM", 
                             #"AdversarialDeepFool"
                    ], help="query strategy")
args = parser.parse_args()
pprint(vars(args))
print()

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset = get_dataset(args.dataset_name)                   # load dataset
net = get_net(args.dataset_name, device)                   # load network
strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

# start experiment
dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

# round 0 accuracy
print("Round 0")
strategy.train()
preds = strategy.predict(dataset.get_test_data())
print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")

for rd in range(1, args.n_round+1):
    print(f"Round {rd}")

    # query
    query_idxs = strategy.query(args.n_query)

    # update labels
    strategy.update(query_idxs)
    strategy.train()

    # calculate accuracy
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")
    
### NEW ###
def plot_decision_boundary(net, dataset, device, title):
    # Define the grid range based on the data range
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5

    # Create a grid of points
    x1 = np.linspace(x_min, x_max, 100)
    x2 = np.linspace(y_min, y_max, 100)
    x1, x2 = np.meshgrid(x1, x2)
    Xtest = np.c_[x1.ravel(), x2.ravel()]
    
    Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32).to(device)

    with torch.no_grad():
        net.clf.eval()
        out, _ = net.clf(Xtest_tensor)
        yest = out.max(1)[1].cpu().numpy()

    yest = yest.reshape(x1.shape)

    # Map labels back to -1 and 1 for visualization
    y_train_mapped = np.where(dataset.Y_train == 1, 1, -1)

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Plot the decision boundary
    ax[0].contourf(x1, x2, yest, alpha=0.3)
    ax[0].scatter(dataset.X_train[:, 0], dataset.X_train[:, 1], c=dataset.Y_train, edgecolor='k', s=20)
    ax[0].set_title(title)
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)

    # Plot the original dataset
    pos = np.where(y_train_mapped == 1)
    neg = np.where(y_train_mapped == -1)
    ax[1].scatter(dataset.X_train[pos, 0], dataset.X_train[pos, 1], c='c', edgecolor='k', label='Class 1')
    ax[1].scatter(dataset.X_train[neg, 0], dataset.X_train[neg, 1], c='m', edgecolor='k', label='Class -1')
    ax[1].set_title('Original Spiral Dataset')
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    ax[1].legend()
    ax[1].set_xlim(x_min, x_max)
    ax[1].set_ylim(y_min, y_max)

    plt.show()

# Plot initial decision boundary
plot_decision_boundary(net, dataset, device, "Decision Boundary")
