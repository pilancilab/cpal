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
parser.add_argument('--dataset_name', type=str, default="Spiral", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "Spiral", "Quadratic"], help="dataset")
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
# obtain tesst/train rmse versus data budget graph
rmse_test_list = []
rmse_train_list = []

# start experiment
selected_indices.extend(dataset.initialize_labels(args.n_init_labeled))
# print(selected_indices)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

# round 0 accuracy
print("Round 0")
strategy.train()
test_preds = strategy.predict(dataset.get_test_data())
# print(len(test_preds)) # 20
_, my_train_data = dataset.get_train_data()
train_preds = strategy.predict(my_train_data)
# print(len(train_preds)) # 80
all_preds = strategy.predict(dataset.get_all_data())
# print(len(all_preds)) # 100

# round = 0
rmse_test_list.append(dataset.cal_rmse_test(test_preds))
rmse_train_list.append(dataset.cal_rmse_train(train_preds))
    
# print(f"Round 0 rmse on test set: {dataset.cal_rmse_test(test_preds)}; rmse on train set: {dataset.cal_rmse_train(train_preds)}; rmse overall: {dataset.cal_rmse_all(all_preds)}; r2 on test set: {dataset.cal_r2_test(test_preds)}; r2 on train set: {dataset.cal_r2_train(train_preds)}; r2 overall: {dataset.cal_r2_all(all_preds)}.")



for rd in range(1, args.n_round+1):
    print(f"Round {rd}")

    # query
    query_idxs = strategy.query(args.n_query)
    if isinstance(query_idxs, (list, np.ndarray)):
        selected_indices.extend(query_idxs)  # Extend if multiple indices are returned
    else:
        selected_indices.append(query_idxs)  # Append if a single index is returned

    # selected_indices.append(query_idxs) # append for EntropySampling, extend for R.S.

    # update labels
    strategy.update(query_idxs)
    strategy.train()

    # evaluate performance
    test_preds = strategy.predict(dataset.get_test_data()) # somehow 80 which is train data
    _, my_train_data = dataset.get_train_data()
    train_preds = strategy.predict(my_train_data)
    all_preds = strategy.predict(dataset.get_all_data())
    
    # print(f"Round {rd} rmse on test set: {dataset.cal_rmse_test(test_preds)}; rmse on train set: {dataset.cal_rmse_train(train_preds)}; rmse overall: {dataset.cal_rmse_all(all_preds)}; r2 on test set: {dataset.cal_r2_test(test_preds)}; r2 on train set: {dataset.cal_r2_train(train_preds)}; r2 overall: {dataset.cal_r2_all(all_preds)}.")
    
    rmse_test_list.append(dataset.cal_rmse_test(test_preds))
    rmse_train_list.append(dataset.cal_rmse_train(train_preds))
    
### NEW ###

def plot_regression(net, dataset, selected_indices, device):
    # Define the grid range
    x_min, x_max = -1, 1
    
    # Create a grid of points for smooth curve plotting
    x_vals = np.linspace(x_min, x_max, 100)
    y_true = x_vals**2
    Xtest = np.column_stack((x_vals, np.ones_like(x_vals)))  # Add bias term for regression

    Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32).to(device)

    with torch.no_grad():
        net.clf.eval()  # Put the network in evaluation mode
        # y_pred = net(Xtest_tensor).cpu().numpy()  # Predict regression values for the test points
        # y_pred = net.predict(Xtest_tensor).cpu().numpy() 
        out, _ = net.clf(dataset.X_all)
        # out, _ = net.clf(Xtest_tensor)
        y_pred = out.cpu().numpy() # this seems to be wrong... it is so off the correct value # think it is because this package is written for classificatio task but not regression tasks so many things need to be modified.
    
    # print(y_pred)
    strat_name = strategy.get_name()
    # Plot the true regression curve (e.g., y = x^2) and predicted curve
    # fig, ax = plt.subplots(figsize=(7, 7))
    plt.figure(figsize=(8, 8))
    # ax.plot(x_vals, y_true, color = 'green', label='True Curve: y = x^2', linewidth=2)
    
    #plt.plot(x_vals, y_true, 'k-', label='True y = x^2')

    # Plot selected data and test data
    
    # First, select the rows with selected_indices, then remove the bias column (i.e., the last column)
    X_selected = dataset.X_train[selected_indices]  # Select the rows using selected_indices
    X_selected_no_bias = X_selected[:, :-1]  # Remove the bias column

    # Now plot the selected data
   
    #plt.scatter(X_selected_no_bias, dataset.Y_train[selected_indices], color='blue', label=f'Queried Points ({strat_name})', s=50)
    #plt.scatter(dataset.X_test[:, :-1], dataset.Y_test, color='red', label='Test Data', alpha=0.5, marker='x')

    # Plot predicted curve
    # plt.plot(x_vals, y_pred, color='cyan', label='Predicted Curve', linewidth=2)
    plt.plot(x_vals, y_pred, label=f'Prediction ({strat_name})', linewidth=2)

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Active Learning ({strat_name})') # TODO: add strategy name on the title
    plt.legend()
    plt.savefig(f'{strat_name}.pdf', bbox_inches='tight')
    plt.show()

    
# Plot learned regression
plot_regression(net, dataset, selected_indices, device)
print(f'selected indices: {selected_indices}')
print(f'rmse train: {rmse_train_list}')
print(f'rmse test: {rmse_test_list}')


# Save RMSE lists using pickle
strategy_name = args.strategy_name

# with open(f'rmse_{strategy_name}_test50_seed4_complete.pkl', 'wb') as f:
#     pickle.dump(rmse_test_list, f)

# with open(f'rmse_{strategy_name}_train50_seed4_complete.pkl', 'wb') as f:
#     pickle.dump(rmse_train_list, f)

