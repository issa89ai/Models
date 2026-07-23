import sys
import torch
import torchvision.transforms as transforms
sys.path.insert(0, '../image-classifier-pipeline')
from voc_dataloader import VocDataset, VOC_CLASSES
from classifier import SimpleClassifier
from src.resnet_yolo import resnet50
from src.config import VOC_CLASSES as YOLO_CLASSES, YOLO_IMG_DIM

data_path = "../image-classifier-pipeline/data/VOCdevkit/VOC2007"

# Load one real VOC image, two different ways (each model expects a different input size)
classifier_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])
yolo_transform = transforms.Compose([
    transforms.Resize((YOLO_IMG_DIM, YOLO_IMG_DIM)),
    transforms.ToTensor(),
])

classifier_dataset = VocDataset(data_path, "val", classifier_transform)
yolo_dataset = VocDataset(data_path, "val", yolo_transform)

classifier_image, classifier_label = classifier_dataset[0]
yolo_image, _ = yolo_dataset[0]

classifier = SimpleClassifier()
yolo_model = resnet50(pretrained=False)

classifier.eval()
yolo_model.eval()

with torch.no_grad():
    classifier_output = classifier(classifier_image.unsqueeze(0))
    yolo_output = yolo_model(yolo_image.unsqueeze(0))

print("=== Image Classifier ===")
print(f"Input shape:  {tuple(classifier_image.shape)}  (1 image, 3 color channels, 227x227 pixels)")
print(f"Output shape: {tuple(classifier_output.shape)}  (1 image, 21 class scores)")
print(f"Meaning: one confidence score per class for the WHOLE image -- 'is a chair present anywhere?'")

print()
print("=== YOLO Object Detector ===")
print(f"Input shape:  {tuple(yolo_image.shape)}  (1 image, 3 color channels, 448x448 pixels)")
print(f"Output shape: {tuple(yolo_output.shape)}  (1 image, {yolo_output.shape[1]}x{yolo_output.shape[2]} grid, 30 values per cell)")
print(f"Meaning: the image is divided into a {yolo_output.shape[1]}x{yolo_output.shape[2]} grid; EACH cell independently")
print(f"predicts 2 bounding boxes (x, y, w, h, confidence = 5 values x 2 = 10) plus")
print(f"20 class probabilities = 30 values -- 'is there an object centered HERE, where")
print(f"exactly is its box, and what class is it?'")

print()
print(f"Classifier total output values: {classifier_output.numel()}")
print(f"YOLO total output values: {yolo_output.numel()}")
