import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
# Fix locale encoding issue
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle # for saving lists

# handle complaints
import os
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="Spiral", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "Spiral"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="EntropySampling", 
                    choices=["RandomSampling", 
                             "LeastConfidence", 
                             #"MarginSampling", 
                             "EntropySampling", 
                             #"LeastConfidenceDropout", 
                             #"MarginSamplingDropout", 
                             #"EntropySamplingDropout", 
                             "KMeansSampling",
                             #"KCenterGreedy", 
                             "BALDDropout", 
                             "AdversarialBIM", 
                             "GreedySampling",
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

selected_indices = []
# obtain test/train acc versus data budget graph
acc_test_list = []
acc_train_list = []

# start experiment
selected_indices.extend(dataset.initialize_labels(args.n_init_labeled))
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

# round 0 accuracy
print("Round 0")
strategy.train()
preds = strategy.predict(dataset.get_test_data())
test_acc = dataset.cal_test_acc(preds)
acc_test_list.append(test_acc)
print(f"Round 0 testing accuracy: {test_acc}")

_, my_train_data = dataset.get_train_data()
train_preds = strategy.predict(my_train_data)
train_acc = dataset.cal_train_acc(train_preds)
acc_train_list.append(train_acc)
print(f"Round 0 training accuracy: {train_acc}")



for rd in range(1, args.n_round+1):
    print(f"Round {rd}")

    # query
    query_idxs = strategy.query(args.n_query)
    if isinstance(query_idxs, (list, np.ndarray)):
        selected_indices.extend(query_idxs)  # Extend if multiple indices are returned
    else:
        selected_indices.append(query_idxs)  # Append if a single index is returned

    # update labels
    strategy.update(query_idxs)
    strategy.train()

    # calculate accuracy
    preds = strategy.predict(dataset.get_test_data())
    test_acc = dataset.cal_test_acc(preds)
    acc_test_list.append(test_acc)
    print(f"Round {rd} testing accuracy: {test_acc}")
    
    _, my_train_data = dataset.get_train_data()
    train_preds = strategy.predict(my_train_data)
    train_acc = dataset.cal_train_acc(train_preds)
    acc_train_list.append(train_acc)
    print(f"Round {rd} training accuracy: {train_acc}")
    
### NEW ###

def plot_decision_boundary(net, dataset, device):
    # Define the grid range based on the data range
    x_min, x_max = -1.5, 1.5 # 1.5
    y_min, y_max = -1.5, 1.5

    # Create a grid of points
    x1 = np.linspace(x_min, x_max, 100)
    x2 = np.linspace(y_min, y_max, 100)
    x1, x2 = np.meshgrid(x1, x2)
    Xtest = np.c_[x1.ravel(), x2.ravel()]
    Xtest = np.append(Xtest, np.ones((Xtest.shape[0], 1)), axis=1)  # Add the bias term
    
    Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32).to(device)

    with torch.no_grad():
        net.clf.eval()
        out, _ = net.clf(Xtest_tensor)
        yest = out.max(1)[1].cpu().numpy()

    yest = yest.reshape(x1.shape)

    # Map labels back to -1 and 1 for visualization
    y_train_mapped = np.where(dataset.Y_train == 1, 1, -1)
    y_test_mapped = np.where(dataset.Y_test == 1, 1, -1)
    
    X_selected = dataset.X_train[selected_indices]
    y_selected = y_train_mapped[selected_indices]

    # Create subplots
    fig, ax = plt.subplots(figsize=(7, 7))

    # Define the custom colors
    colors = ['#920783', '#00b7c7']  # Switched the colors to match the image
    cmap = mcolors.ListedColormap(colors)

    # Plot the decision boundary with custom colors
    ax.contourf(x1, x2, yest, alpha=0.3, cmap=cmap)
    scatter_train = ax.scatter(dataset.X_train[:, 0], dataset.X_train[:, 1], c=y_train_mapped, edgecolor='k', s=20, cmap=cmap,
                               label='Train Data')
    scatter_test = ax.scatter(dataset.X_test[:, 0], dataset.X_test[:, 1], c=y_test_mapped, edgecolor='k', s=20, cmap=cmap,
                              marker='^', label = 'Test Data')
    scatter_select = ax.scatter(X_selected[:,0], X_selected[:,1], c=y_selected, s=80, cmap=cmap, marker='x',
                               label='Queried Data')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'{args.strategy_name}')
    plt.legend()
    plt.savefig(f'{args.strategy_name}.pdf', bbox_inches='tight')
    plt.show()
    
# Save accuracy lists using pickle
strategy_name = args.strategy_name  
# Plot initial decision boundary
plot_decision_boundary(net, dataset, device)

# with open(f'acc_{strategy_name}_test50_seed0_complete.pkl', 'wb') as f:
#     pickle.dump(acc_test_list, f)

# with open(f'acc_{strategy_name}_train50_seed0_complete.pkl', 'wb') as f:
#     pickle.dump(acc_train_list, f)
