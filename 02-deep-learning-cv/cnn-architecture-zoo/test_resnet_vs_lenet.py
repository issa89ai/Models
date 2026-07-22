import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from models.lenet import LeNet
from models.resnet import ResNet18

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_subset = Subset(trainset, range(5000))
test_subset = Subset(testset, range(1000))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(model, name, epochs=10):
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


lenet = LeNet()
resnet18 = ResNet18()

lenet_acc = train_and_evaluate(lenet, "LeNet", epochs=10)
resnet_acc = train_and_evaluate(resnet18, "ResNet18", epochs=10)

print(f"Summary: LeNet = {lenet_acc:.4f}, ResNet18 = {resnet_acc:.4f}")
