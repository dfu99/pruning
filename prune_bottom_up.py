import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from os.path import isfile, join

# Set random seed for reproducibility
torch.manual_seed(42)

# File index for saving every iteration and not overwriting files
def get_last_file_index(filename):
    path = os.path.join("data", "bottom_up_pruning") # Data directory to save results
    all_files = [file for file in os.listdir(path) if isfile(join(path, file)) and file.startswith("history")]
    if len(all_files) == 0:
        return 0
    last_file_idx = all_files[-1].split("_")[-1].replace(".txt", "")
    return int(last_file_idx)+1

# Download and prepare MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split into train and test sets (80/20)
train_size = int(0.8 * len(mnist_data))
test_size = len(mnist_data) - train_size
train_dataset, test_dataset = random_split(mnist_data, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define a flexible neural network
class FlexibleNet(nn.Module):
    def __init__(self, hidden_layers):
        super(FlexibleNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(28*28, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        self.layers.append(nn.Linear(hidden_layers[-1], 10))
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

# Training function
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    for batch_idx, (X, Y) in enumerate(train_loader):
        x, y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, Y in test_loader:
            x, y = X.to(device), Y.to(device)
            output = model(x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

# Main experiment
def run_experiment(max_neurons, idx, mode="progressive"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracies = []
    configurations = []
    # Iterate through different hidden layer shapes
    for num_layers in range(1, 9):
        for neurons in range(1, max_neurons+1):
            if mode=="progressive":
                hidden_layers = [max_neurons] * (num_layers-1) + [neurons]
            elif mode == "uniform":
                hidden_layers = [neurons] * num_layers
            model = FlexibleNet(hidden_layers).to(device)
            optimizer = optim.Adam(model.parameters())
            loss_fn = nn.CrossEntropyLoss()

            # Train each instance for a fixed number of epochs
            for epoch in range(5):
                train(model, train_loader, optimizer, loss_fn, device)

            # Evaluate the model for accuracy
            accuracy = evaluate(model, test_loader, device)
            accuracies.append(accuracy)
            
            # Output the testing history
            configurations.append(f"{num_layers} layers, with dimensions {hidden_layers}")
            status_string = f"Configuration: {configurations[-1]}, Accuracy: {accuracy:.4f}"
            config_filename = join("data", "bottom_up_pruning", f"history_{idx:03d}.txt")
            with open(config_filename, 'a') as f:
                f.write(status_string)
                f.write("\n")
            print(status_string)

    return accuracies, configurations

# Run the experiment
max_neurons = 16
for mode in ["progressive", "uniform"]:
    idx = get_last_file_index("history")
    accuracies, configurations = run_experiment(max_neurons, idx, mode)

    # Plot the results
    plt.figure(figsize=(15, 6))
    plt.plot(accuracies)
    plt.title("Accuracy vs Network Complexity")
    plt.xlabel("Configuration")
    plt.ylabel("Accuracy")
    plt.xticks(range(0, len(configurations), 8), [configurations[i] for i in range(0, len(configurations), 8)], rotation=45, ha='right')
    plt_filename = join("data", "bottom_up_pruning", mode, f'accuracy_{idx:03d}.png')
    plt.savefig(plt_filename, bbox_inches='tight')
    # plt.show()

    print(f"Experiment completed. Results plotted and saved as '{plt_filename}'.")