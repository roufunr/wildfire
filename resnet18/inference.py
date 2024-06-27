import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import time

# Set the paths
model_path = './saved_model/wildfire_classifier_resnet18.pth'
image_path = '/home/rouf/wildfire_dataset/test/Smoke/Smoke (1).jpeg'  # Replace with the path to your image

# Define transforms for the image
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model
model = models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 3)  # Assuming 3 classes
)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load and preprocess the image
image = Image.open(image_path)
image = image_transforms(image).unsqueeze(0)  # Add batch dimension

# Run inference
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = model.to(device)
image = image.to(device)

with torch.no_grad():
    start = time.time() * 1000
    outputs = model(image)
    end = time.time() * 1000
    _, preds = torch.max(outputs, 1)

# Define the class names (replace with your actual class names)
class_names = ['fire', 'no fire', 'smoke']

# Print the predicted class
print(f'Predicted class: {class_names[preds.item()]} total time: {end - start}')
