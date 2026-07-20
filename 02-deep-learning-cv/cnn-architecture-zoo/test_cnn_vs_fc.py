import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from models.lenet import LeNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use a subset for a fast CPU-friendly demo
train_subset = Subset(trainset, range(5000))
test_subset = Subset(testset, range(1000))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)


class FullyConnectedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the image -- throws away spatial structure
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(model, name, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"{name} - epoch {epoch + 1}/{epochs} done, last batch loss: {loss.item():.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"{name}: test accuracy = {accuracy:.4f}, parameters = {count_params(model):,}\n")
    return accuracy


fc_net = FullyConnectedNet()
lenet = LeNet()

fc_acc = train_and_evaluate(fc_net, "Fully Connected Net", epochs=10)
cnn_acc = train_and_evaluate(lenet, "LeNet (CNN)", epochs=10)

print(f"Summary: FC net = {fc_acc:.4f}, LeNet (CNN) = {cnn_acc:.4f}")
