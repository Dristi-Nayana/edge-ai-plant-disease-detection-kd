# %%
!pip install kaggle tqdm


# %%
import os

dataset_root = "/content/PlantVillage/Plant_leave_diseases_dataset_without_augmentation"  # Check if this path is correct

# List subfolders (each should be a class label)
class_folders = [folder for folder in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, folder))]

# Count the number of classes
num_classes = len(class_folders)

print(f"✅ Number of classes: {num_classes}")
print(f"✅ Class labels: {class_folders}")


# %%
import os
import matplotlib.pyplot as plt
from PIL import Image

# Path to your dataset
dataset_root = "/content/PlantVillage/Plant_leave_diseases_dataset_without_augmentation"

# List class folders
class_folders = [folder for folder in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, folder))]
class_folders.sort()  # Optional: Sort alphabetically

# Parameters for visualization
images_per_class = 3  # Number of images to show per class
img_size = (128, 128)  # Resize for visualization
cols = images_per_class
rows = len(class_folders)

# Set up plot
fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
fig.suptitle("📊 PlantVillage Dataset Preview", fontsize=16)

# Loop through each class and show images
for row_idx, class_name in enumerate(class_folders):
    class_path = os.path.join(dataset_root, class_name)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))][:images_per_class]

    for col_idx in range(images_per_class):
        ax = axs[row_idx, col_idx] if rows > 1 else axs[col_idx]
        if col_idx < len(image_files):
            img_path = os.path.join(class_path, image_files[col_idx])
            img = Image.open(img_path).resize(img_size)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')

        # Label only the first image in each row with class name
        if col_idx == 0:
            ax.set_ylabel(class_name, rotation=0, size='large', labelpad=70)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# %%
import os
import matplotlib.pyplot as plt
from PIL import Image
import random

# Path to your dataset
dataset_root = "/content/PlantVillage/Plant_leave_diseases_dataset_without_augmentation"

# Get all class folders
class_folders = [folder for folder in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, folder))]
class_folders.sort()

# Choose 5 classes randomly
selected_classes = random.sample(class_folders, 5)

# Parameters
images_per_class = 5  # 5 images from each of 5 classes
img_size = (128, 128)

# Setup the grid
fig, axs = plt.subplots(5, 5, figsize=(15, 15))
fig.suptitle("🌿 PlantVillage Dataset: 5x5 Sample Grid", fontsize=20)

# Loop through selected classes
for row_idx, class_name in enumerate(selected_classes):
    class_path = os.path.join(dataset_root, class_name)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    selected_images = random.sample(image_files, min(5, len(image_files)))

    for col_idx in range(5):
        ax = axs[row_idx, col_idx]
        if col_idx < len(selected_images):
            img_path = os.path.join(class_path, selected_images[col_idx])
            img = Image.open(img_path).resize(img_size)
            ax.imshow(img)
            ax.set_title(f"{class_name}" if col_idx == 0 else "", fontsize=8)
        ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %%
import os
import matplotlib.pyplot as plt
from PIL import Image
import random

# Center crop function
def center_crop(image, crop_size=(100, 100)):
    width, height = image.size
    new_width, new_height = crop_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return image.crop((left, top, right, bottom))

# Dataset path
dataset_root = "/content/PlantVillage/Plant_leave_diseases_dataset_without_augmentation"
class_folders = [folder for folder in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, folder))]
class_folders.sort()

# Randomly pick 5 classes
selected_classes = random.sample(class_folders, 5)

# Plot
fig, axs = plt.subplots(5, 5, figsize=(12, 12))
fig.suptitle("Zoomed-in Diseased Leaf Regions (Center Cropped)", fontsize=18)

for row_idx, class_name in enumerate(selected_classes):
    class_path = os.path.join(dataset_root, class_name)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    selected_images = random.sample(image_files, min(5, len(image_files)))

    for col_idx in range(5):
        ax = axs[row_idx, col_idx]
        if col_idx < len(selected_images):
            img_path = os.path.join(class_path, selected_images[col_idx])
            img = Image.open(img_path).convert('RGB')
            cropped_img = center_crop(img, crop_size=(100, 100))
            ax.imshow(cropped_img)
            ax.set_title(class_name if col_idx == 0 else "", fontsize=7)
        ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %% [markdown]
# # **preprocessing**

# %%
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load dataset
dataset = datasets.ImageFolder(root=dataset_root, transform=transform)

# Split dataset into training (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"✅ Preprocessing done!")
print(f"📌 Training samples: {len(train_dataset)}")
print(f"📌 Validation samples: {len(val_dataset)}")
print(f"📌 Total classes: {len(dataset.classes)}")


# %%
import matplotlib.pyplot as plt
from collections import Counter

# Count number of images per class
class_counts = Counter([dataset.classes[label] for _, label in dataset.samples])

# Sort classes alphabetically for a consistent plot
sorted_classes = sorted(class_counts.keys())
sorted_counts = [class_counts[cls] for cls in sorted_classes]

# Plot the distribution
plt.figure(figsize=(14, 6))
plt.bar(sorted_classes, sorted_counts, color="skyblue")
plt.xticks(rotation=90)
plt.xlabel("Class Labels")
plt.ylabel("Number of Images")
plt.title("Class Distribution in the Dataset")
plt.show()

# Print class-wise count
for cls, count in zip(sorted_classes, sorted_counts):
    print(f"{cls}: {count} images")


# %%
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Define transformations (basic preprocessing + augmentation for minority classes)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images randomly
    transforms.RandomRotation(20),  # Rotate images slightly
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust colors
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load dataset
dataset = ImageFolder(root=dataset_root, transform=transform)

# Compute class weights for Weighted Sampling
class_counts = np.array([class_counts[dataset.classes[i]] for i in range(len(dataset.classes))])
class_weights = 1.0 / class_counts  # Inverse class frequency
sample_weights = [class_weights[label] for _, label in dataset.samples]

# Use WeightedRandomSampler for balanced sampling
sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)

# DataLoader with Weighted Sampling
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)

print("✅ Preprocessing and balancing done! Ready for model training.")


# %%
import matplotlib.pyplot as plt
import numpy as np

# Function to denormalize images for visualization
def denormalize(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.numpy().transpose((1, 2, 0))  # Convert to HWC format
    image = image * std + mean  # Denormalize
    image = np.clip(image, 0, 1)  # Clip values between 0 and 1
    return image

# Get a batch of preprocessed images
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Plot a few images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()
for i in range(10):
    img = denormalize(images[i])
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(dataset.classes[labels[i]])

plt.tight_layout()
plt.show()


# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the dataset directory
dataset_root = "/content/PlantVillage/Plant_leave_diseases_dataset_without_augmentation"

# Define transformations for training and testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Function to load the dataset when needed
def get_dataloader(batch_size=32, shuffle=True, num_workers=2):
    dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, dataset.classes  # Returns dataloader and class names

# Load the dataset when needed
train_loader, class_names = get_dataloader(batch_size=32)

# Print class names and dataset size
print(f"✅ {len(class_names)} classes loaded.")
print(f"✅ Dataset contains {len(train_loader.dataset)} images.")


# %%
import matplotlib.pyplot as plt
import numpy as np

# Function to visualize a batch of images
def visualize_batch(dataloader, class_names, num_images=16):
    images, labels = next(iter(dataloader))  # Get a batch of images
    images = images[:num_images]  # Select only required images
    labels = labels[:num_images]  # Select corresponding labels

    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean  # Reverse normalization
    images = images.numpy().transpose(0, 2, 3, 1)  # Convert to (batch, H, W, C)

    # Plot images
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)  # 4x4 grid
        plt.imshow(images[i])
        plt.title(class_names[labels[i].item()])
        plt.axis("off")

    plt.show()

# Call function to visualize a batch
visualize_batch(train_loader, class_names, num_images=16)


# %% [markdown]
# # **model training**

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MobileViT model (pretrained on ImageNet)
model = timm.create_model("mobilevit_s", pretrained=True, num_classes=39)
model = model.to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        # Training loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        val_acc = evaluate_model(model, val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

# Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100 * correct / total

# Start Training
train_model(model, train_loader, val_loader, epochs=10)


# %%
import torch
import timm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MobileViT model
model = timm.create_model("mobilevit_s", pretrained=True, num_classes=39)
model = model.to(device)



# %%
!pip install timm # install timm

# %%
# Save trained model
torch.save(model.state_dict(), "mobilevit_trained.pth")
print("✅ Model saved successfully as 'mobilevit_trained.pth'.")


# %%
import torch
import timm

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = timm.create_model("mobilevit_s", pretrained=False, num_classes=39)
model = model.to(device)

# Load saved weights
model.load_state_dict(torch.load("mobilevit_trained.pth", map_location=device))
model.eval()  # Set to evaluation mode

print("✅ Model loaded successfully!")


# %%
# Define the class labels (Replace with your actual class names)
class_names = ['Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Cherry___healthy', 'Tomato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Peach___healthy', 'Cherry___Powdery_mildew', 'Soybean___healthy', 'Corn___healthy', 'Corn___Northern_Leaf_Blight', 'Apple___Apple_scab', 'Tomato___Early_blight', 'Peach___Bacterial_spot', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Raspberry___healthy', 'Grape___Esca_(Black_Measles)', 'Tomato___Bacterial_spot', 'Tomato___Leaf_Mold', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Grape___healthy', 'Blueberry___healthy', 'Potato___healthy', 'Squash___Powdery_mildew', 'Apple___Black_rot', 'Tomato___Target_Spot', 'Tomato___Septoria_leaf_spot', 'Potato___Early_blight', 'Pepper,_bell___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Background_without_leaves', 'Pepper,_bell___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Grape___Black_rot', 'Tomato___healthy', 'Potato___Late_blight', 'Corn___Common_rust', 'Strawberry___Leaf_scorch', 'Strawberry___healthy']  # Ensure correct order

# Get class name
predicted_label = class_names[predicted_class]
print(f"Predicted class: {predicted_label}")


# %%
top3_prob, top3_classes = torch.topk(probabilities, 3)
top3_prob = top3_prob.cpu().numpy().flatten()
top3_classes = top3_classes.cpu().numpy().flatten()

print("Top 3 Predictions:")
for i in range(3):
    print(f"{class_names[top3_classes[i]]}: {top3_prob[i] * 100:.2f}%")


# %%
import os

image_folder = "/content/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/Strawberry___Leaf_scorch"
image_files = [f for f in os.listdir(image_folder) if f.endswith(".JPG")]

for image_file in image_files[:5]:  # Test first 5 images
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    print(f"{image_file} -> Predicted class: {class_names[predicted_class]}")


# %%
import matplotlib.pyplot as plt

# Accuracy values from your training log
epochs = list(range(1, 11))
train_acc = [93.89, 97.65, 98.12, 98.34, 98.46, 98.70, 98.69, 98.96, 98.92, 98.89]
val_acc = [94.45, 97.11, 99.03, 98.17, 98.70, 99.54, 98.28, 97.85, 99.26, 98.85]

# Plot the accuracy curve
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, marker='o', linestyle='-', color='blue', label='Train Accuracy')
plt.plot(epochs, val_acc, marker='s', linestyle='--', color='red', label='Validation Accuracy')

# Labels and title
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training vs. Validation Accuracy")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# %%
# Loss values from training log
train_loss = [0.2312, 0.0734, 0.0583, 0.0526, 0.0470, 0.0395, 0.0399, 0.0339, 0.0321, 0.0343]
val_loss = [None, None, None, None, None, None, None, None, None, None]  # Fill in val_loss if available

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', linestyle='-', color='blue', label='Train Loss')
plt.plot(epochs, val_loss, marker='s', linestyle='--', color='red', label='Validation Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


# %%
# Replace with your actual learning rate values
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.00001, 0.00001, 0.00001]


plt.figure(figsize=(8, 5))
plt.plot(epochs, learning_rates, marker='o', linestyle='-', color='purple', label='Learning Rate')

plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True)
plt.show()


# %%
import torch
import matplotlib.pyplot as plt

# Select an image from the validation set
image, _ = next(iter(val_loader))  # Get a single batch
image = image[0].unsqueeze(0).to(device)  # Select one image and add batch dimension

# Get feature maps from intermediate layers
feature_maps = []
hooks = []

# Function to extract feature maps
def hook_fn(module, input, output):
    feature_maps.append(output)

# Register hooks to the convolutional layers of MobileViT
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Conv2d):  # Only extract feature maps from Conv layers
        hooks.append(layer.register_forward_hook(hook_fn))

# Forward pass through the model
with torch.no_grad():
    _ = model(image)

# Remove hooks after extraction
for hook in hooks:
    hook.remove()


# %%
def plot_feature_maps(feature_maps, num_cols=8):
    for i, fmap in enumerate(feature_maps):
        fmap = fmap.squeeze(0).cpu()  # Remove batch dimension & move to CPU
        num_filters = fmap.shape[0]  # Number of filters (channels)

        num_rows = (num_filters + num_cols - 1) // num_cols  # Calculate rows needed
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2))

        for j in range(num_filters):
            row, col = j // num_cols, j % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.imshow(fmap[j], cmap="viridis")
            ax.axis("off")

        plt.suptitle(f"Feature Maps from Layer {i+1}", fontsize=14)
        plt.show()

# Plot extracted feature maps
plot_feature_maps(feature_maps)


# %%
import torch

def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# %%
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import timm  # Import timm to load models

# Define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimized KD Loss Function
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=5.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kd = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets (Distillation Loss)
        soft_targets = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = self.criterion_kd(soft_targets, teacher_soft_targets)

        # Hard targets (CrossEntropy Loss)
        classification_loss = self.criterion_ce(student_logits, labels)

        # Weighted sum of both losses
        return (1 - self.alpha) * classification_loss + self.alpha * distillation_loss

# Optimized Training Function
def train_kd_model(teacher, student, train_loader, val_loader, epochs=10, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Move models to device
    teacher.to(device)
    student.to(device)

    # Freeze teacher model
    teacher.eval()

    optimizer = optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    kd_loss_fn = KnowledgeDistillationLoss(alpha=0.7, temperature=5.0)

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Get Teacher & Student Predictions
            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)

            # Compute KD Loss
            loss = kd_loss_fn(student_logits, teacher_logits, labels)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        val_acc = evaluate_model(student, val_loader)

        # Adjust learning rate
        scheduler.step(running_loss / len(train_loader))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    print("Training complete!")

# Load Pretrained Teacher Model (MobileViT)
teacher_model = timm.create_model("mobilevit_s", pretrained=True, num_classes=39)
teacher_model.eval().to(device)  # Set to evaluation mode

# Define Student Model (Smaller Version of Teacher)
student_model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=39)
student_model.to(device)

# Start KD Training
train_kd_model(teacher_model, student_model, train_loader, val_loader, epochs=10)


# %%
def train_kd_model(teacher, student, train_loader, val_loader, epochs=10, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Move models to device
    teacher.to(device)
    student.to(device)
    teacher.eval()  # Freeze teacher

    # Optimizer and scheduler
    optimizer = optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    kd_loss_fn = KnowledgeDistillationLoss(alpha=0.7, temperature=5.0)

    # Data for visualization
    history = {
        'train_accuracy': [],
        'val_accuracy': [],
        'total_loss': [],
        'ce_loss': [],
        'kd_loss': []
    }

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_kd_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)

            # Distillation parts
            soft_targets = nn.functional.log_softmax(student_logits / kd_loss_fn.temperature, dim=1)
            teacher_soft = nn.functional.softmax(teacher_logits / kd_loss_fn.temperature, dim=1)

            kd_loss = kd_loss_fn.criterion_kd(soft_targets, teacher_soft)
            ce_loss = kd_loss_fn.criterion_ce(student_logits, labels)
            total_loss = (1 - kd_loss_fn.alpha) * ce_loss + kd_loss_fn.alpha * kd_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += total_loss.item()
            running_ce_loss += ce_loss.item()
            running_kd_loss += kd_loss.item()

            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        val_acc = evaluate_model(student, val_loader)
        scheduler.step(running_loss / len(train_loader))

        # Save for visualization
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['total_loss'].append(running_loss / len(train_loader))
        history['ce_loss'].append(running_ce_loss / len(train_loader))
        history['kd_loss'].append(running_kd_loss / len(train_loader))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {history['total_loss'][-1]:.4f}, "
              f"CE Loss: {history['ce_loss'][-1]:.4f}, KD Loss: {history['kd_loss'][-1]:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    print("Training complete!")
    return history


# %%
import torch
import timm  # Import timm if not already imported

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your student model architecture (e.g., MobileNetV3)
student_model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=39)

# Load the saved state dictionary (if you trained it previously)
try:
    student_model.load_state_dict(torch.load("student_model.pth", map_location=device))
    print("Loaded pre-trained student model weights.")
except FileNotFoundError:
    print("Pre-trained student model weights not found. Starting with a new model.")

# Now you can save the model
torch.save(student_model.state_dict(), "student_model.pth")
print("Student model saved as 'student_model.pth'")

# %%
import torch
import timm

# 1. Load the Model Architecture:
student_model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=39)

# 2. Load the Saved State Dictionary:
student_model.load_state_dict(torch.load("student_model.pth"))

# 3. Move to Device (if using GPU):
student_model.to(device)

# 4. Set to Evaluation Mode:
student_model.eval()

print("✅ Student model loaded successfully!")

# %%
import os

def get_model_size_mb(model_path):
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)

# Example:
teacher_model_path = '/content/mobilevit_trained.pth'  # replace with your actual model file
student_model_path = '/content/student_model.pth'

print("Teacher Model Size (MB):", get_model_size_mb(teacher_model_path))
print("Student Model Size (MB):", get_model_size_mb(student_model_path))


# %%
student_model.load_state_dict(torch.load("student_model.pth"))
student_model.to(device)
student_model.eval()


# %%
import matplotlib.pyplot as plt

def plot_kd_history(history):
    epochs = range(1, len(history['train_accuracy']) + 1)

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Total Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['total_loss'], label='Total KD Loss', color='blue')
    plt.plot(epochs, history['ce_loss'], label='CrossEntropy Loss', color='green')
    plt.plot(epochs, history['kd_loss'], label='KL Distillation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses during Knowledge Distillation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Assuming 'train_kd_model', 'teacher_model', 'student_model', 'train_loader', and 'val_loader' are defined
history = train_kd_model(teacher_model, student_model, train_loader, val_loader, epochs=10) # Call train_kd_model and assign the returned value to history

# Now call the plotting function
plot_kd_history(history)

# %% [markdown]
# ## without **kd**

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a smaller student model (e.g., MobileNetV3)
student_model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=39)
student_model = student_model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# Training function (same style as teacher)
def train_student_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        val_acc = evaluate_model(model, val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

# Reuse your evaluation function from earlier
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100 * correct / total

# Train student model
train_student_model(student_model, train_loader, val_loader, epochs=10)


# %%
def train_model(model, train_loader, val_loader, epochs=10):
    history = {
        'train_accuracy': [],
        'val_accuracy': [],
        'loss': []
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        val_acc = evaluate_model(model, val_loader)

        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(running_loss / len(train_loader))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    return history


# %%
import matplotlib.pyplot as plt

def plot_teacher_vs_student(teacher_history, student_history):
    epochs = range(1, len(teacher_history['train_accuracy']) + 1)

    # Accuracy Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, teacher_history['train_accuracy'], label='Teacher Train Acc', linestyle='--')
    plt.plot(epochs, teacher_history['val_accuracy'], label='Teacher Val Acc')
    plt.plot(epochs, student_history['train_accuracy'], label='Student Train Acc', linestyle='--')
    plt.plot(epochs, student_history['val_accuracy'], label='Student Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy: Teacher vs Student')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Loss Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, teacher_history['loss'], label='Teacher Loss')
    plt.plot(epochs, student_history['loss'], label='Student Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss: Teacher vs Student')
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
%matplotlib inline


# %%
def plot_losses(history):
    epochs = range(1, len(history['total_loss']) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['total_loss'], label='Total KD Loss', color='blue')
    plt.plot(epochs, history['ce_loss'], label='CrossEntropy Loss', color='green')
    plt.plot(epochs, history['kd_loss'], label='KL Divergence Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Components During KD Training')
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
def plot_accuracy_gap(history, teacher_acc=99.5):
    epochs = range(1, len(history['val_accuracy']) + 1)
    student_acc = history['val_accuracy']
    gap = [teacher_acc - acc for acc in student_acc]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, gap, label='Accuracy Gap (Teacher - Student)', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Gap in Accuracy (%)')
    plt.title('Knowledge Transfer Effectiveness')
    plt.grid(True)
    plt.legend()
    plt.show()


# %%
def plot_accuracy_gap(history, teacher_acc=100.0):
    epochs = range(1, len(history['train_accuracy']) + 1)
    acc_gap = [teacher_acc - acc for acc in history['val_accuracy']]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc_gap, color='purple', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Gap (%)')
    plt.title('Gap Between Teacher and Student Accuracy (Val)')
    plt.grid(True)
    plt.show()


# %%
import timm
import torch

# Load Pre-trained MobileViT Model
teacher_model = timm.create_model('mobilevit_xs', pretrained=True)
teacher_model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)

print("MobileViT model loaded successfully!")


# %%
import timm

# Load Pre-trained MobileViT
teacher_model = timm.create_model("mobilevit_xs", pretrained=True)
teacher_model.eval()


# %%
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

def extract_features(model, dataloader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            output = model.forward_features(x)  # works for timm models
            features.append(output.cpu())
            labels.extend(y.cpu())
    return torch.cat(features).numpy(), labels

# Get features
teacher_features, teacher_labels = extract_features(teacher_model, val_loader)
student_features, student_labels = extract_features(student_model, val_loader)

# Reduce dimensions
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
teacher_2d = tsne.fit_transform(teacher_features)
student_2d = tsne.fit_transform(student_features)

# Plot
def plot_tsne(data, labels, title):
    df = pd.DataFrame()
    df["x1"], df["x2"] = data[:, 0], data[:, 1]
    df["label"] = labels
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="x1", y="x2", hue="label", palette="tab10", data=df)
    plt.title(title)
    plt.show()

plot_tsne(teacher_2d, teacher_labels, "Teacher Model Feature Space")
plot_tsne(student_2d, student_labels, "Student Model Feature Space")


# %%
def plot_softmax_distribution(model, image_tensor, title):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        softmax = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(softmax)), softmax)
    plt.title(title)
    plt.xlabel("Class Index")
    plt.ylabel("Softmax Confidence")
    plt.show()

# Compare for a sample image
plot_softmax_distribution(teacher_model, image, "Teacher Model - Softmax Output")
plot_softmax_distribution(student_model, image, "Student Model - Softmax Output")


# %%
plt.plot(train_acc_list, label="Train Acc (Student)")
plt.plot(val_acc_list, label="Val Acc (Student)")
plt.title("Knowledge Distillation Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()


# %%
import torch
import timm  # Assuming you're using timm for model loading

# Load Pretrained Teacher Model (MobileViT)
teacher_model = timm.create_model("mobilevit_s", pretrained=True, num_classes=39)

# If you have a specific device (e.g., GPU), move the model to that device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)

# Now you can print the model summary
print(teacher_model)

# %%


# %%


# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# Knowledge Distillation Loss Function
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        soft_targets = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        soft_labels = nn.functional.softmax(teacher_logits / self.temperature, dim=1)

        kd_loss = self.kl_div(soft_targets, soft_labels) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, true_labels)

        return self.alpha * ce_loss + (1 - self.alpha) * kd_loss

# Load Pre-trained Teacher Model (MobileViT)
teacher_model = torch.hub.load('apple/ml-mobilevit', 'mobilevit_xs')
teacher_model.eval()  # No training for the teacher

# Define Student Model (Smaller version of MobileViT)
student_model = models.mobilenet_v2(num_classes=10)  # Adjust for your dataset

# Move models to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)

# Define Optimizer and Scheduler
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Initialize Loss Function
kd_loss_fn = KnowledgeDistillationLoss(alpha=0.7, temperature=4.0)

# Training Function
def train_kd_model(teacher, student, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Get Teacher & Student Predictions
            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)

            # Compute Knowledge Distillation Loss
            loss = kd_loss_fn(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = student_logits.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        scheduler.step()  # Adjust learning rate

        # Compute Validation Accuracy
        val_acc = evaluate_model(student, val_loader)
        train_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

# Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# Feature Map Visualization (Grad-CAM)
def visualize_feature_maps(model, images):
    model.eval()
    images = images.to(device)

    def hook_fn(module, input, output):
        global feature_maps
        feature_maps = output

    handle = model.features[0][0].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(images)  # Run forward pass to capture feature maps

    handle.remove()  # Remove hook after visualization

    feature_maps = feature_maps.cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(10, 5))

    for i in range(4):
        axes[i].imshow(feature_maps[0, i], cmap='viridis')
        axes[i].axis('off')

    plt.show()

# Example: Visualizing Feature Maps from Student Model
sample_images, _ = next(iter(train_loader))
visualize_feature_maps(student_model, sample_images[:1])


# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Get features from Teacher & Student models
def extract_features(model, dataloader):
    model.eval()
    features_list, labels_list = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model.forward_features(images)  # Feature extraction layer
            features_list.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    return np.vstack(features_list), np.hstack(labels_list)

# Extract features
teacher_features, labels = extract_features(teacher_model, val_loader)
student_features, _ = extract_features(student_model, val_loader)

# Reduce dimensions using PCA
pca = PCA(n_components=50)
teacher_pca = pca.fit_transform(teacher_features)
student_pca = pca.fit_transform(student_features)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
teacher_tsne = tsne.fit_transform(teacher_pca)
student_tsne = tsne.fit_transform(student_pca)

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(teacher_tsne[:, 0], teacher_tsne[:, 1], c=labels, cmap='jet', alpha=0.7)
plt.title("Teacher Model Feature Clusters")

plt.subplot(1, 2, 2)
plt.scatter(student_tsne[:, 0], student_tsne[:, 1], c=labels, cmap='jet', alpha=0.7)
plt.title("Student Model Feature Clusters")

plt.show()


# %%
import seaborn as sns

def plot_logits_distribution(model, dataloader, title):
    model.eval()
    logits_list = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            logits_list.append(probs)

    logits_array = np.vstack(logits_list)
    sns.histplot(logits_array.flatten(), bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Softmax Probability")
    plt.ylabel("Frequency")
    plt.show()

plot_logits_distribution(teacher_model, val_loader, "Teacher Model Logits Distribution")
plot_logits_distribution(student_model, val_loader, "Student Model Logits Distribution")


# %%


# %%


# %%


# %%


# %%
import torch

# Extract feature embeddings before classification
def extract_features(model, images):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        features = model.forward_features(images)  # Get embeddings
    return features.cpu().numpy()

# Example usage
sample_images, _ = next(iter(val_loader))  # Get batch of validation images
features = extract_features(model, sample_images)
print("Extracted Feature Shape:", features.shape)  # Check dimensions


# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming `features` is a NumPy array of shape (batch_size, 640, 7, 7)
batch_size, channels, h, w = features.shape

# Flatten the spatial dimensions (7x7) and take mean across them
features_flat = np.mean(features.reshape(batch_size, channels, -1), axis=2)

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_flat)

# Apply PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Generate random labels (for visualization only) - Replace with actual class labels if available
labels = np.random.randint(0, 39, size=batch_size)

# Plot the PCA embeddings
plt.figure(figsize=(10, 6))
scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap="viridis", alpha=0.7)
plt.colorbar(scatter, label="Class Labels")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Extracted Features")
plt.show()


# %%
import torch
import time

def measure_inference_time(model, input_size=(1, 3, 224, 224), device='cuda', repeat=100):
    model.eval()
    model.to(device)

    dummy_input = torch.randn(input_size).to(device)

    # Warm-up (important for accurate GPU timing)
    for _ in range(10):
        _ = model(dummy_input)

    start = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(dummy_input)
    end = time.time()

    avg_time_ms = ((end - start) / repeat) * 1000
    return round(avg_time_ms, 2)


# %%
teacher_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=39)


# %%
import torch
import timm
import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Teacher Model
teacher_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=39)
teacher_model.load_state_dict(torch.load('/content/mobilevit_trained.pth', map_location=device))
teacher_model.to(device)
teacher_model.eval()

# Load Student Model (if also trained for 39 classes)
student_model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=39)
student_model.load_state_dict(torch.load('/content/student_model.pth', map_location=device))
student_model.to(device)
student_model.eval()

# Dummy input for measuring inference time
dummy_input = torch.randn(1, 3, 224, 224).to(device)

def measure_inference_time(model, input_tensor, runs=100):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    end = time.time()
    avg_time = (end - start) / runs * 1000  # ms
    return round(avg_time, 2)

# Measure
teacher_time = measure_inference_time(teacher_model, dummy_input)
student_time = measure_inference_time(student_model, dummy_input)

print(f"✅ Teacher Inference Time: {teacher_time} ms")
print(f"✅ Student Inference Time: {student_time} ms")


# %%
def calculate_accuracy_retention(teacher_acc, student_acc):
    return round((student_acc / teacher_acc) * 100, 2)

# Example (replace these with your real values)
teacher_accuracy = 98.89  # in percentage
student_accuracy = 98.69  # in percentage

accuracy_retention = calculate_accuracy_retention(teacher_accuracy, student_accuracy)
print(f"✅ Accuracy Retention: {accuracy_retention}%")


# %%
import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['MobileViT (Teacher)', 'MobileNetV3 (Student)']
accuracies = [98.89, 98.69]  # Replace with actual values
model_sizes = [19.1, 6.07]  # In MB

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots()

# Accuracy bar (left axis)
ax1.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='skyblue')
ax1.set_ylabel('Accuracy (%)', color='skyblue')
ax1.set_ylim(0, 100)
ax1.tick_params(axis='y', labelcolor='skyblue')

# Model size bar (right axis)
ax2 = ax1.twinx()
ax2.bar(x + width/2, model_sizes, width, label='Model Size (MB)', color='salmon')
ax2.set_ylabel('Model Size (MB)', color='salmon')
ax2.tick_params(axis='y', labelcolor='salmon')

# Labels & Legends
plt.xticks(x, models, rotation=15)
fig.suptitle('Accuracy vs. Model Size')
fig.tight_layout()
plt.show()


# %%
# Data
inference_times = [15.13, 7.32]  # ms
accuracy_retention = [100, 99.8]  # % of teacher accuracy

# Plot
plt.figure(figsize=(8,5))
plt.plot(inference_times, accuracy_retention, marker='o', linestyle='-', color='purple')

# Annotate points
for x, y in zip(inference_times, accuracy_retention):
    plt.text(x, y+0.2, f"{y}%", ha='center')

plt.title('Inference Time vs. Accuracy Retention')
plt.xlabel('Inference Time (ms)')
plt.ylabel('Accuracy Retention (%)')
plt.grid(True)
plt.ylim(95, 101)
plt.show()


# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data for teacher and student models
model_names = ['Teacher (MobileViT)', 'Student (MobileNetV3)', 'Ideal Goal']
model_sizes = [19.1, 6.07, 1.91]            # MB
inference_times = [15.13, 7.32, 5]          # ms (example ideal inference time)
accuracy_drop = [0.0, 0.2, 10]              # % drop (0 for teacher, 0.2 for student, 10 for ideal)

colors = ['blue', 'green', 'red']
markers = ['o', '^', '*']

# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(model_names)):
    ax.scatter(model_sizes[i], inference_times[i], accuracy_drop[i],
               color=colors[i], marker=markers[i], s=100, label=model_names[i])
    ax.text(model_sizes[i], inference_times[i], accuracy_drop[i] + 0.5,
            f"{model_names[i]}", fontsize=9, color=colors[i])

# Axes labels
ax.set_xlabel('Model Size (MB)')
ax.set_ylabel('Inference Time (ms)')
ax.set_zlabel('Accuracy Drop (%)')
ax.set_title('📌 Benchmark: Accuracy Drop vs. Model Size vs. Inference Time')

# Customize view
ax.view_init(elev=20, azim=45)
ax.legend()
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Colors
teacher_color = "#66c2a5"
student_color = "#fc8d62"
arrow_color = "black"

# Draw teacher and student blocks
teacher = patches.FancyBboxPatch((0.1, 0.6), 0.25, 0.3, boxstyle="round,pad=0.02",
                                 edgecolor='black', facecolor=teacher_color, label='Teacher')
student = patches.FancyBboxPatch((0.6, 0.6), 0.25, 0.3, boxstyle="round,pad=0.02",
                                 edgecolor='black', facecolor=student_color, label='Student')

ax.add_patch(teacher)
ax.add_patch(student)

# Labels
ax.text(0.225, 0.9, "Teacher Model", ha="center", fontsize=12, fontweight='bold')
ax.text(0.725, 0.9, "Student Model", ha="center", fontsize=12, fontweight='bold')

# Arrows and KD methods
# 1. Logit-based KD (top)
ax.annotate("Logit-Based KD\n(Soft Label Transfer)", xy=(0.35, 0.75), xytext=(0.55, 0.75),
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2),
            ha='center', va='center', fontsize=10)

# 2. Feature-based KD (middle)
ax.annotate("Feature-Based KD\n(Intermediate Activations)", xy=(0.35, 0.65), xytext=(0.55, 0.65),
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2),
            ha='center', va='center', fontsize=10)

# 3. Attention-based KD (bottom)
ax.annotate("Attention-Based KD\n(Spatial Attention Maps)", xy=(0.35, 0.55), xytext=(0.55, 0.55),
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2),
            ha='center', va='center', fontsize=10)

# Title
plt.title("📌 Knowledge Distillation Methods", fontsize=14, fontweight='bold')

# Show plot
plt.tight_layout()
plt.show()


# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Simulate a dummy input tensor (like an image)
input_tensor = torch.rand(1, 3, 32, 32)  # batch x channel x H x W

# --- 1. Logit-Based KD ---
teacher_logits = torch.rand(1, 10) * 3
student_logits = torch.rand(1, 10) * 2

teacher_soft = F.softmax(teacher_logits, dim=1).detach().numpy().flatten()
student_soft = F.softmax(student_logits, dim=1).detach().numpy().flatten()

plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.bar(range(10), teacher_soft, color='blue')
plt.title("Teacher Soft Labels")
plt.subplot(1, 2, 2)
plt.bar(range(10), student_soft, color='orange')
plt.title("Student Soft Labels")
plt.suptitle("Logit-Based KD")
plt.tight_layout()
plt.show()

# --- 2. Feature-Based KD ---
teacher_feature = torch.rand(16, 8, 8)  # Simulated feature map
student_feature = torch.rand(16, 8, 8)

# Select 1 channel for comparison
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(teacher_feature[0].detach().numpy(), cmap='viridis')
plt.title("Teacher Feature")
plt.subplot(1, 2, 2)
plt.imshow(student_feature[0].detach().numpy(), cmap='plasma')
plt.title("Student Feature")
plt.suptitle("Feature-Based KD")
plt.tight_layout()
plt.show()

# --- 3. Attention-Based KD ---
# Simulate spatial attention maps (normalized)
def get_attention_map(feature_map):
    attention = feature_map.mean(0)  # mean over channels
    attention = (attention - attention.min()) / (attention.max() - attention.min())  # normalize
    return attention

teacher_attn = get_attention_map(teacher_feature)
student_attn = get_attention_map(student_feature)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(teacher_attn.numpy(), cmap='hot')
plt.title("Teacher Attention")
plt.subplot(1, 2, 2)
plt.imshow(student_attn.numpy(), cmap='coolwarm')
plt.title("Student Attention")
plt.suptitle("Attention-Based KD")
plt.tight_layout()
plt.show()


# %%
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create teacher and student models
teacher_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=39)
student_model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=39)

# Load weights
def load_weights(model, path):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model

teacher_model = load_weights(teacher_model, '/content/mobilevit_trained.pth').to(device).eval()
student_model = load_weights(student_model, '/content/student_model.pth').to(device).eval()


# %%
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create teacher and student models
teacher_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=39)
student_model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=39)

# Load weights
def load_weights(model, path):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model

teacher_model = load_weights(teacher_model, '/content/mobilevit_trained.pth').to(device).eval()
student_model = load_weights(student_model, '/content/student_model.pth').to(device).eval()

teacher_features = {}
student_features = {}

def get_feature_hook(name, storage_dict):
    def hook(module, input, output):
        storage_dict[name] = output
    return hook

# Get the correct layers for hook registration
# Find the names of the desired layers using model.named_modules() or model._modules
# Example:
for name, module in teacher_model.named_modules():
    print(name)
for name, module in student_model.named_modules():
    print(name)

# Find the desired layers' names and use them below
teacher_layer_name = "stages.11.blocks.1.mlp.fc2"  # Replace with the correct teacher layer name
student_layer_name = "blocks.15.conv"  # Replace with the correct student layer name

# Register hooks using the correct layer names
teacher_layer = teacher_model.get_submodule(teacher_layer_name)
student_layer = student_model.get_submodule(student_layer_name)

teacher_layer.register_forward_hook(get_feature_hook("feat", teacher_features))
student_layer.register_forward_hook(get_feature_hook("feat", student_features))

# %%
teacher_model.head


# %%
student_model.classifier

# %%
student_model.global_pool

# %%
pip install timm matplotlib seaborn scikit-learn opencv-python


# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utility Functions ---

def preprocess_image(img_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def register_feature_hook(model, layer, storage):
    def hook(module, input, output):
        storage['features'] = output.detach()
    return layer.register_forward_hook(hook)

def show_feature_map_comparison(feature_dict, title):
    fmap = feature_dict['features'].squeeze(0).cpu()
    fmap = fmap[:min(4, fmap.shape[0])]
    fig, axs = plt.subplots(1, len(fmap), figsize=(12, 3))
    for i in range(len(fmap)):
        axs[i].imshow(fmap[i], cmap='viridis')
        axs[i].axis('off')
        axs[i].set_title(f"Channel {i}")
    plt.suptitle(title)
    plt.show()

def apply_grad_cam(model, image, target_layer):
    image.requires_grad_()
    gradients = {}
    activations = {}

    def forward_hook(module, input, output):
        activations["value"] = output
    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(image)
    class_idx = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients["value"].squeeze(0).cpu().numpy()
    acts = activations["value"].squeeze(0).cpu().numpy()
    weights = np.mean(grads, axis=(1, 2))
    cam = np.sum(weights[:, None, None] * acts, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_fw.remove()
    handle_bw.remove()
    return cam

def overlay_heatmap(cam, image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return overlay

# --- Main Execution ---

img_path = "/content/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/Grape___Black_rot/image (100).JPG"
img = preprocess_image(img_path)

# 1. Logits Comparison
teacher_logits = teacher_model(img)
student_logits = student_model(img)

plt.figure(figsize=(10, 4))
plt.plot(teacher_logits.squeeze().detach().cpu().numpy(), label="Teacher", marker='o')
plt.plot(student_logits.squeeze().detach().cpu().numpy(), label="Student", marker='x')

plt.title("Logits Comparison")
plt.xlabel("Class Index")
plt.ylabel("Logit Value")
plt.legend()
plt.grid(True)
plt.show()

# 2. t-SNE Feature Comparison
teacher_feat, student_feat = {}, {}
teacher_hook = register_feature_hook(teacher_model, teacher_model.head.global_pool, teacher_feat)
student_hook = register_feature_hook(student_model, student_model.conv_stem, student_feat)

_ = teacher_model(img)
_ = student_model(img)
t_feat = teacher_feat['features'].view(teacher_feat['features'].size(0), -1).cpu().numpy()

s_feat = student_feat['features'].squeeze().flatten(1).cpu().numpy()

features = np.vstack([t_feat, s_feat])
labels = ['Teacher'] * len(t_feat) + ['Student'] * len(s_feat)

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_result = tsne.fit_transform(features)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette=['blue', 'orange'])
plt.title("t-SNE Clustering: Teacher vs Student Features")
plt.show()

teacher_hook.remove()
student_hook.remove()

# 3. Feature Map Visualization
show_feature_map_comparison(teacher_feat, "Teacher Feature Maps")
show_feature_map_comparison(student_feat, "Student Feature Maps")

# 4. Grad-CAM Visualization
teacher_cam = apply_grad_cam(teacher_model, img.clone(), teacher_model.head.global_pool)
student_cam = apply_grad_cam(student_model, img.clone(), student_model.blocks[-1])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(overlay_heatmap(teacher_cam, img_path))
plt.title("Teacher Grad-CAM")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay_heatmap(student_cam, img_path))
plt.title("Student Grad-CAM")
plt.axis("off")
plt.show()


# %%
pip install onnx onnxruntime torchvision


# %%
pip install torch torchvision timm onnx onnxruntime torch-tensorboard


# %%
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models import mobilenet_v3_large
import torch.quantization
import torch.nn.intrinsic.quantized as nnq

# 1. Load MobileNetV3 and modify for 39 classes
mobilenetv3_model = mobilenet_v3_large(weights=None)
mobilenetv3_model.classifier[3] = nn.Linear(1024, 39)

# 2. Load trained weights
mobilenetv3_model.load_state_dict(torch.load('student_model.pth', map_location='cpu'))
mobilenetv3_model.eval()

# 3. Apply dynamic quantization (suitable for conv nets)
quant_mobilenetv3 = torch.quantization.quantize_dynamic(
    mobilenetv3_model, {nn.Linear}, dtype=torch.qint8
)

# 4. Apply pruning
parameters_to_prune = []
for module_name, module in quant_mobilenetv3.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

# Prune 30% of weights using L1 unstructured
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3
)

# Remove pruning reparametrizations
for module, name in parameters_to_prune:
    prune.remove(module, name)

# 5. Export to ONNX
def export_onnx(model, file_name):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust input size if needed
    torch.onnx.export(
        model,
        dummy_input,
        file_name,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )

# Export compressed model
export_onnx(quant_mobilenetv3, "mobilenetv3_compressed.onnx")
print("Exported pruned and quantized MobileNetV3 to ONNX.")


# %%
import torch
import timm

# Define model architecture and load weights
student_model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=39)
student_model.load_state_dict(torch.load("student_model.pth", map_location="cpu"))
student_model.eval()


# %%
import torch.nn.utils.prune as prune
import torch.nn as nn

def apply_pruning(model, amount=0.3):  # Prune 30% of weights
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model

pruned_model = apply_pruning(student_model, amount=0.3)
torch.save(pruned_model.state_dict(), "student_pruned.pth")


# %%
quantized_model = torch.quantization.quantize_dynamic(
    student_model, {nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), "student_quantized.pth")


# %%
pip install onnx-tf


# %%
!pip uninstall tensorflow-addons

# %%
!pip install tensorflow-addons --no-deps

# %%
!pip install keras

# %%
!pip install tensorflow==2.11.0
!pip install tensorflow-addons==0.19.0
!pip uninstall keras -y
!pip install keras==2.11.0


# %%
!pip install tensorflow-addons

# %%
!pip install tensorflow-addons keras

# %%
pip install tflite-runtime


# %%
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# Load interpreter
interpreter = tflite.Interpreter(model_path="mobilenetv3_pruned_quantized.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
img = Image.open("/content/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/Apple___Apple_scab/image (1).JPG").resize((224, 224))
img_array = np.array(img).astype(np.float32) / 255.0
img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
img_array = np.expand_dims(img_array, axis=0)

# Set input
interpreter.set_tensor(input_details[0]['index'], img_array)

# Run inference
interpreter.invoke()

# Get prediction
pred = interpreter.get_tensor(output_details[0]['index'])
print("Prediction:", np.argmax(pred))


# %%


# %%
train_datagen = ImageDataGenerator(preprocessing_function=lambda img: train_transform(Image.fromarray(img)).numpy())

# %%
!pip install tensorflow==2.12.0
!pip install -U tensorflow-model-optimization

# %%
!pip install tensorflow-model-optimization

# %%
import torch.nn.utils.prune as prune
import torch.nn as nn

# Apply pruning to the student model
def apply_pruning(model, amount=0.3):  # Remove 30% of the connections
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

# Load the trained model
student_model.load_state_dict(torch.load("student_model.pth", map_location=device))
pruned_model = apply_pruning(student_model)

# Save pruned model
torch.save(pruned_model.state_dict(), "student_model_pruned.pth")
print("✅ Pruned model saved.")


# %%
import torch.nn as nn
from torch.quantization import quantize_dynamic
import torch.nn.utils.prune as prune # Import prune
import timm

# Apply pruning to the student model
def apply_pruning(model, amount=0.3):  # Remove 30% of the connections
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount) # Apply Pruning using prune.l1_unstructured
    return model

# Load the correct model architecture (MobileViT)
student_model = timm.create_model("mobilevit_s", pretrained=False, num_classes=39)
student_model.to(device)

# Load the trained model without pruning applied during saving
student_model.load_state_dict(torch.load("mobilevit_trained.pth", map_location=device)) # Load the trained MobileViT model

# Apply pruning and quantization, or clone if you want to keep original for comparison
pruned_model = apply_pruning(student_model) # Apply Pruning
quantize_dynamic(pruned_model, {nn.Linear}, dtype=torch.qint8, inplace=True) # Apply Quantization in-place on the pruned model

# Save pruned and quantized model
torch.save(pruned_model.state_dict(), "student_model_quantized.pth")
print("✅ Pruned and Quantized model saved.")

# %%
!pip install tensorflow==2.12.0
!pip install -U tensorflow-model-optimization
!pip install timm
!pip install grad-cam


import torch
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm
from copy import deepcopy  # Import deepcopy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the dataset directory
dataset_root = "/content/PlantVillage/Plant_leave_diseases_dataset_without_augmentation"

# Define transformations for training and testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Load the correct model architecture (MobileViT)
student_model = timm.create_model("mobilevit_s", pretrained=False, num_classes=39)
student_model.to(device)

# Load the trained model without pruning applied during saving
student_model.load_state_dict(torch.load("mobilevit_trained.pth", map_location=device))

# Create a deepcopy of the model before pruning
model_to_export = deepcopy(student_model)

# Remove pruning to allow successful ONNX conversion
for name, module in model_to_export.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        try:
            prune.remove(module, 'weight')  # Remove pruning masks if they exist
        except ValueError:
            pass  # If pruning hasn't been applied, ignore ValueError

# Define the class labels (Replace with your actual class names)
class_names = ['Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Cherry___healthy', 'Tomato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Peach___healthy', 'Cherry___Powdery_mildew', 'Soybean___healthy', 'Corn___healthy', 'Corn___Northern_Leaf_Blight', 'Apple___Apple_scab', 'Tomato___Early_blight', 'Peach___Bacterial_spot', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Raspberry___healthy', 'Grape___Esca_(Black_Measles)', 'Tomato___Bacterial_spot', 'Tomato___Leaf_Mold', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Grape___healthy', 'Blueberry___healthy', 'Potato___healthy', 'Squash___Powdery_mildew', 'Apple___Black_rot', 'Tomato___Target_Spot', 'Tomato___Septoria_leaf_spot', 'Potato___Early_blight', 'Pepper,_bell___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Background_without_leaves', 'Pepper,_bell___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Grape___Black_rot', 'Tomato___healthy', 'Potato___Late_blight', 'Corn___Common_rust', 'Strawberry___Leaf_scorch', 'Strawberry___healthy']  # Ensure correct order

# Export to ONNX using model_to_export
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    model_to_export,  # Export the unpruned copy
    dummy_input,
    "student_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
print("✅ Model exported to ONNX format.")

# %%
pip install tensorflow==2.12.0 tf2onnx onnx


# %%
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm
from copy import deepcopy  # Import deepcopy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
import torch.utils.model_zoo as model_zoo
from torch import prune

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the dataset directory
dataset_root = "/content/PlantVillage/Plant_leave_diseases_dataset_without_augmentation"

# Define transformations for training and testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Load the correct model architecture (MobileViT)
student_model = timm.create_model("mobilevit_s", pretrained=False, num_classes=39)
student_model.to(device)

# Load the trained model without pruning applied during saving
try:
    student_model.load_state_dict(torch.load("mobilevit_trained.pth", map_location=device))
    print("✅ Trained model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Create a deepcopy of the model before pruning
model_to_export = deepcopy(student_model)

# Ensure the model is in evaluation mode
model_to_export.eval()

# Remove pruning to allow successful ONNX conversion
for name, module in model_to_export.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        try:
            prune.remove(module, 'weight')  # Remove pruning masks if they exist
            print(f"Pruning removed from layer: {name}")
        except ValueError:
            print(f"No pruning mask found for layer: {name}")
            pass  # If pruning hasn't been applied, ignore ValueError

# Define the class labels (Replace with your actual class names)
class_names = [
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Cherry___healthy', 'Tomato___Late_blight',
    'Tomato___Tomato_mosaic_virus', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Peach___healthy',
    'Cherry___Powdery_mildew', 'Soybean___healthy', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
    'Apple___Apple_scab', 'Tomato___Early_blight', 'Peach___Bacterial_spot',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Raspberry___healthy', 'Grape___Esca_(Black_Measles)',
    'Tomato___Bacterial_spot', 'Tomato___Leaf_Mold', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Grape___healthy', 'Blueberry___healthy', 'Potato___healthy', 'Squash___Powdery_mildew',
    'Apple___Black_rot', 'Tomato___Target_Spot', 'Tomato___Septoria_leaf_spot', 'Potato___Early_blight',
    'Pepper,_bell___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Background_without_leaves',
    'Pepper,_bell___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Grape___Black_rot',
    'Tomato___healthy', 'Potato___Late_blight', 'Corn___Common_rust', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy'
]

# Ensure the input size is correct
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Try exporting the model to ONNX format
try:
    torch.onnx.export(
        model_to_export,  # Export the unpruned copy
        dummy_input,
        "student_model.onnx",
        export_params=True,
        opset_version=13,  # Use a newer opset version
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        verbose=True
    )
    print("✅ Model exported to ONNX format successfully.")
except Exception as e:
    print(f"❌ Error during ONNX export: {e}")


# %%
!pip install torch --upgrade


# %%
import torch
import torch.nn as nn
import torch.onnx
import torch.quantization as quant
import timm

# Load and prepare the student model (MobileViT)
student_model = timm.create_model("mobilevit_s", pretrained=False, num_classes=39)
student_model.load_state_dict(torch.load("student_model.pth", map_location=device))
student_model.to(device)

# Apply pruning (30% pruning for instance)
def apply_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

pruned_model = apply_pruning(student_model)

# Apply quantization (dynamic quantization in this case)
quantize_dynamic(pruned_model, {nn.Linear}, dtype=torch.qint8, inplace=True)

# Export the pruned and quantized model to ONNX format
dummy_input = torch.randn(1, 3, 224, 224).to(device)
onnx_model_path = "student_model_quantized.onnx"
torch.onnx.export(pruned_model, dummy_input, onnx_model_path, input_names=['input'], output_names=['output'], opset_version=13)
print(f"✅ Model exported to ONNX format: {onnx_model_path}")

# Now, use ONNX Runtime on Raspberry Pi to load and run inference


# %%
!pip install torchvision --upgrade

# %%
!pip install --upgrade torchvision torchaudio torchtext

# %%
import torch
import torch.nn as nn
import timm

# Load MobileNetV3 student model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=39)
model.load_state_dict(torch.load("student_model.pth", map_location=device))
model.to(device)
model.eval()


# %%













































import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make pruning permanent
    return model

model = apply_pruning(model, amount=0.3)


# %%
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


# %%
!pip install onnx # install onnx package

# %% [markdown]
# # onxx

# %%
dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(
    quantized_model,
    dummy_input,
    "student_model_pruned_quantized.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    export_params=True,
    opset_version=12
)

print("✅ ONNX model exported: student_model_pruned_quantized.onnx")


# %%
import onnxruntime as ort
import numpy as np

# ✅ Load the ONNX model
def load_onnx_model(model_path="student_model_pruned_quantized.onnx"):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

# ✅ Run inference with a single input image (preprocessed)
def run_inference(session, input_name, output_name, input_tensor):
    # input_tensor: shape (1, 3, 224, 224), dtype float32
    outputs = session.run([output_name], {input_name: input_tensor})
    predicted_class = np.argmax(outputs[0])
    return predicted_class


# %%
!pip install onnxruntime # install onnxruntime
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("student_model_pruned_quantized.onnx")
input_name = session.get_inputs()[0].name

# Dummy input (numpy)
dummy_input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Inference
output = session.run(None, {input_name: dummy_input_np})
print("✅ ONNX inference ran successfully.")

# %%
from google.colab import files
files.download("student_model_pruned_quantized.onnx")


# %%
!ls -lh *.pth
!ls -lh *.onnx


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



