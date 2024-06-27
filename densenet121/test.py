import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Set the paths
dataset_base_path = '/home/rouf/wildfire_dataset'
test_data_dir = f'{dataset_base_path}/test'
model_path = './saved_model/wildfire_classifier_densenet121.pth'

# Define transforms for testing
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
test_dataset = datasets.ImageFolder(test_data_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model
model = models.densenet121()
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, len(test_dataset.classes))
)
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')

# Additional metrics
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
