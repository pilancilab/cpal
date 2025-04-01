import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import locale
from deepal_baseline.utils import get_dataset, get_net, get_strategy
import pickle

# Fix locale encoding + MKL issue
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_init_labeled', type=int, default=10000)
    parser.add_argument('--n_query', type=int, default=1000)
    parser.add_argument('--n_round', type=int, default=10)
    parser.add_argument('--dataset_name', type=str, default="Spiral",
                        choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "Spiral", "Quadratic"])
    parser.add_argument('--strategy_name', type=str, default="EntropySampling",
                        choices=["RandomSampling", "LeastConfidence", "EntropySampling",
                                 "KMeansSampling", "BALDDropout", "AdversarialBIM"])
    parser.add_argument('--save_dir', type=str, default="plots")
    parser.add_argument('--save_pickle', action='store_true')
    return parser.parse_args()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False

def run_active_learning(args):
    setup_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = get_dataset(args.dataset_name)
    net = get_net(args.dataset_name, device)
    strategy = get_strategy(args.strategy_name)(dataset, net)

    selected_indices = list(dataset.initialize_labels(args.n_init_labeled))
    print(f"Initial labeled pool: {args.n_init_labeled}")
    print(f"Unlabeled pool: {dataset.n_pool - args.n_init_labeled}")
    print(f"Test pool: {dataset.n_test}\n")

    rmse_train_list = []
    rmse_test_list = []

    # Round 0 evaluation
    strategy.train()
    test_preds = strategy.predict(dataset.get_test_data())
    _, train_data = dataset.get_train_data()
    train_preds = strategy.predict(train_data)
    all_preds = strategy.predict(dataset.get_all_data())

    rmse_train_list.append(dataset.cal_rmse_train(train_preds))
    rmse_test_list.append(dataset.cal_rmse_test(test_preds))

    for rd in range(1, args.n_round + 1):
        print(f"Round {rd}")
        query_idxs = strategy.query(args.n_query)
        selected_indices.extend(query_idxs if isinstance(query_idxs, (list, np.ndarray)) else [query_idxs])

        strategy.update(query_idxs)
        strategy.train()

        test_preds = strategy.predict(dataset.get_test_data())
        _, train_data = dataset.get_train_data()
        train_preds = strategy.predict(train_data)
        all_preds = strategy.predict(dataset.get_all_data())

        rmse_train_list.append(dataset.cal_rmse_train(train_preds))
        rmse_test_list.append(dataset.cal_rmse_test(test_preds))

    return selected_indices, rmse_train_list, rmse_test_list, strategy, dataset, net, device

def plot_regression(net, dataset, selected_indices, strategy, device, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    x_vals = np.linspace(-1, 1, 100)
    y_true = x_vals ** 2
    Xtest = np.column_stack((x_vals, np.ones_like(x_vals)))
    Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32).to(device)

    with torch.no_grad():
        net.clf.eval()
        out, _ = net.clf(Xtest_tensor)
        y_pred = out.cpu().numpy().flatten()

    X_selected = dataset.X_train[selected_indices][:, :-1]
    y_selected = dataset.Y_train[selected_indices]
    strat_name = strategy.get_name()

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_true, 'k--', label='True y = x²')
    plt.plot(x_vals, y_pred, 'r-', label=f'Prediction ({strat_name})', linewidth=2)
    plt.scatter(X_selected, y_selected, color='blue', label='Queried Points', s=50)
    plt.scatter(dataset.X_test[:, :-1], dataset.Y_test, color='gray', alpha=0.4, label='Test Data', marker='x')
    plt.title(f'Active Learning Strategy: {strat_name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{strat_name}.pdf'), bbox_inches='tight')
    plt.show()

def save_rmse_pickle(rmse_train, rmse_test, strategy_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'rmse_{strategy_name}_train.pkl'), 'wb') as f:
        pickle.dump(rmse_train, f)
    with open(os.path.join(save_dir, f'rmse_{strategy_name}_test.pkl'), 'wb') as f:
        pickle.dump(rmse_test, f)

def main():
    args = parse_args()
    selected_indices, rmse_train, rmse_test, strategy, dataset, net, device = run_active_learning(args)
    plot_regression(net, dataset, selected_indices, strategy, device, save_dir=args.save_dir)

    print(f"\nFinal selected indices: {selected_indices}")
    print(f"RMSE train: {rmse_train}")
    print(f"RMSE test: {rmse_test}")

    if args.save_pickle:
        save_rmse_pickle(rmse_train, rmse_test, args.strategy_name, save_dir=args.save_dir)

if __name__ == '__main__':
    main()
