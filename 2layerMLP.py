import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # For plotting
import time

# Hyperparameters
input_size = 28 * 28        # MNIST images are 28x28 pixels
hidden_size = 128           # Number of neurons in the hidden layer
num_classes = 10            # Number of output classes (digits 0-9)
num_epochs = 1              # Number of training epochs
batch_size = 32            # Batch size for training
learning_rate = 0.001       # Learning rate for the optimizer
start = time.time()
# Device configuration (use GPU if available)
device = torch.device('cuda')
#device = torch.device('cpu')
# Define the 1-layer MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()                          # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, num_classes) # Second fully connected layer

    def forward(self, x):
        x = x.view(-1, input_size)  # Flatten the input tensor
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out  # No softmax needed as CrossEntropyLoss includes it

def main():
    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),                      # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST dataset
    ])

    # Download and load the training and test datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )

    # Data loaders for iterating over batches
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Instantiate the model, define the loss function and the optimizer
    model = MLP(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()                       # Cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Lists to keep track of losses and accuracies
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Training loop
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        correct = 0
        total = 0
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: compute predicted outputs
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Update model parameters

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print loss every 100 steps
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Step [{i + 1}/{total_step}], '
                      f'Loss: {loss.item():.4f}')
        print(torch.cuda.is_available())
        print(f'Using device: {device}')
        print(f'Test: {torch.cuda.device_count()}')
        # Calculate average loss and accuracy for the epoch
        train_loss = running_loss / total_step
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate the model on the test dataset
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            correct = 0
            total = 0
            running_loss = 0.0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                # Predicted class is the one with the highest score
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_loss = running_loss / len(test_loader)
            test_accuracy = 100 * correct / total
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Plot training and test accuracy
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'ro-', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, test_losses, 'ro-', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    end = time.time()
    print(end - start)
    # (Optional) Save the trained model to a file
    # torch.save(model.state_dict(), 'mlp_mnist.pth')

if __name__ == '__main__':
    main()
