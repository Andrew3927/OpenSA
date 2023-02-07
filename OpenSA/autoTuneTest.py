import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet

# Initialize Ray.
ray.init()

# Define a training function.
def train(config, reporter):
    # Load the data.
    train_loader, test_loader = get_data_loaders()
    # Initialize the model.
    model = ConvNet()
    # Define the loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                          momentum=config["momentum"])
    # Train the model.
    for epoch in range(config["epochs"]):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Evaluate the model on the test set.
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        # Report the accuracy.
        reporter(mean_accuracy=accuracy)

# Define the search space.
space = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "momentum": tune.uniform(0.1, 0.9),
    "epochs": 10,
}

# Run the experiment.
analysis = tune.run(
    train,
    name="mnist_cnn",
    config=space,
    num_samples=10,
    scheduler=ASHAScheduler(),
    resources_per_trial={
        "cpu": 1,
        "gpu": 0,
    },
    local_dir="./ray_results",
)

# Get the best hyperparameters.
best_config = analysis.get_best_config(metric="mean_accuracy")
print("Best hyperparameters: ", best_config)

# Shut down Ray.
ray.shutdown()
