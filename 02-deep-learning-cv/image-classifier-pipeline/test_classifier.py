import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from voc_dataloader import VocDataset, VOC_CLASSES
from classifier import SimpleClassifier

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])

data_path = "data/VOCdevkit/VOC2007"
train_dataset = VocDataset(data_path, "train", transform)
val_dataset = VocDataset(data_path, "val", transform)

# Small subsets for a fast demo
train_subset = Subset(train_dataset, range(500))
val_subset = Subset(val_dataset, range(200))

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

model = SimpleClassifier()
criterion = nn.BCEWithLogitsLoss()  # multi-label: independent yes/no per class
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 5
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch + 1}/{epochs} done, last batch loss: {loss.item():.4f}")

# Simplified evaluation: per-class accuracy at a 0.5 probability threshold
# (the real assignment uses mean Average Precision -- this is a simpler stand-in)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = torch.sigmoid(model(images))
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.numel()

print(f"Per-class accuracy (0.5 threshold): {correct / total:.4f}")

# Show one real prediction to make the multi-label output concrete
images, labels = next(iter(val_loader))
with torch.no_grad():
    probs = torch.sigmoid(model(images[:1]))
predicted_classes = [VOC_CLASSES[i] for i in range(21) if probs[0, i] > 0.5]
true_classes = [VOC_CLASSES[i] for i in range(21) if labels[0, i] == 1]
print(f"Example -- predicted classes: {predicted_classes}")
print(f"Example -- true classes: {true_classes}")
