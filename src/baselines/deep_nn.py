import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Set seed
torch.manual_seed(0)

# Define deep neural network for regression.
class SimpleRegressor(nn.Module):
    def __init__(self):
        super(SimpleRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),  # 2 input features, including a bias term
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

class DeepBinaryClassifier(nn.Module): # for binary classification
    def __init__(self):
        super(DeepBinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),  # 2 features + 1 bias → 64 hidden
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Output single logit
        )

    def forward(self, x):
        return self.model(x)

# Training function for regression
def train_model(model, x, y, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model

def train_classifier(model, X_train, y_train, epochs=1000, lr=0.01): # training function for binary classification
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, (y_train + 1) / 2)  # Convert -1/+1 → 0/1
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return model

# Plot predictions
def plot_predictions(model, x_tensor_train, y_tensor_train, noise=False):
    """
    Plot DNN regression results.
    
    If `noise=True`, plots:
      - noisy training data
      - predicted y
      - true y = x^2
    
    If `noise=False`, assumes y_tensor_train already reflects the true function.
    
    Args:
        model: trained PyTorch model
        x_tensor_train: torch tensor, shape (n, 2) [feature, bias]
        y_tensor_train: torch tensor, shape (n, 1)
        noise: whether the training data is noisy
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(x_tensor_train).numpy().flatten()

    x_vals = x_tensor_train[:, 0].numpy()  # extract original x values
    y_train = y_tensor_train.numpy().flatten()
    y_true = x_vals ** 2  # actual function y = x^2

    # Sort by x for smooth plotting
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_train_sorted = y_train[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    y_true_sorted = y_true[sort_idx]

    # Plot
    plt.figure(figsize=(8, 4))

    if noise:
        plt.scatter(x_vals, y_train, color='gray', alpha=0.5, label='Training data (noisy)')

    plt.plot(x_sorted, y_true_sorted, 'k--', label='True: y = x²')
    plt.plot(x_sorted, y_pred_sorted, 'r-', linewidth=2, label='DNN Prediction')
    plt.title("DNN Regression on y = x²" + (" (Noisy)" if noise else ""))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_decision_boundary(X, y, X_test, y_test, model, selected_indices, name="DeepNN"):
    """
    Plot the decision boundary of a deep neural network model using PyTorch.
    
    Args:
        X: training data (n_train, 3) → 2 features + bias
        y: training labels (+1 / -1)
        X_test: test data (n_test, 3)
        y_test: test labels (+1 / -1)
        model: trained PyTorch model
        selected_indices: indices in X that were queried (highlighted with 'x')
        name: plot title
    """
    # Define the grid range based on the data range
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5

    # Create mesh grid
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Xgrid = np.c_[x1.ravel(), x2.ravel()]
    bias = np.ones((Xgrid.shape[0], 1))
    Xgrid_input = np.concatenate([Xgrid, bias], axis=1)
    
    # Predict using the model
    model.eval()
    with torch.no_grad():
        Xgrid_tensor = torch.tensor(Xgrid_input, dtype=torch.float32)
        logits = model(Xgrid_tensor).numpy().reshape(x1.shape)

    # Map labels to ±1
    y_train_mapped = np.where(y == 1, 1, -1)
    y_test_mapped = np.where(y_test == 1, 1, -1)
    
    X_selected = X[selected_indices]
    y_selected = y_train_mapped[selected_indices]

    # Custom colormap
    colors = ['#920783', '#00b7c7']
    cmap = mcolors.ListedColormap(colors)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.contourf(x1, x2, logits, alpha=0.3, cmap=cmap)

    # Training data
    ax.scatter(X[:, 0], X[:, 1], c=y_train_mapped, edgecolor='k', s=20, cmap=cmap, label='Train Data')

    # Test data
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test_mapped, edgecolor='k', s=20, cmap=cmap, marker='^', label='Test Data')

    # Queried/selected points
    ax.scatter(X_selected[:, 0], X_selected[:, 1], c=y_selected, s=80, cmap=cmap, marker='x', label='Queried Data')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'{name}')
    plt.legend()
    # plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.show()

# Run everything
if __name__ == "__main__":
    from cpal.synthetic_data import *
    # regression example
    X_all, y_all, X_train, y_train, X_test, y_test = generate_quadratic_regression(
    n=100, n_train=80, seed=0, noise=True, noise_param=0.05, plot=True
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Initialize model
    model = SimpleRegressor()

    # Train
    trained_model = train_model(model, X_train_tensor, y_train_tensor, epochs=1000, lr=0.01)
    plot_predictions(trained_model, X_train_tensor, y_train_tensor, True)

    # binary spiral
    # X_all, y_all, X_train, y_train, X_test, y_test = generate_spiral_data(n=10, n_train=80)
    # model = DeepBinaryClassifier()
    # trained_model = train_classifier(model, X_train, y_train)
    # plot_decision_boundary(X_train, y_train, X_test, y_test, trained_model, selected_indices = [0], name="Spiral DNN")

    