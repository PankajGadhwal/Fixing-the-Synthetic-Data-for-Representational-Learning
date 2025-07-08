import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from PIL import Image
import random

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset paths
real_path = "data/cifake/real"
di_path   = "data/cifake/DI"
sd_path   = "data//cifake/SD"

# Load real dataset
real_dataset = ImageFolder(root=real_path, transform=transform)
real_samples = real_dataset.samples
total = len(real_samples)
indices = list(range(total))
random.shuffle(indices)

# Split into train/val/test
train_size = int(0.7 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_filenames = [os.path.basename(real_samples[i][0]) for i in train_indices]
val_filenames   = [os.path.basename(real_samples[i][0]) for i in val_indices]
test_filenames  = [os.path.basename(real_samples[i][0]) for i in test_indices]

train_real = Subset(real_dataset, train_indices)
val_real   = Subset(real_dataset, val_indices)
test_real  = Subset(real_dataset, test_indices)

# Switch real dataset transform for val/test
real_dataset.transform = test_transform

# Load DI dataset
di_dataset = ImageFolder(root=di_path, transform=transform)
di_samples = di_dataset.samples
di_name_to_index = {os.path.basename(p[0]): i for i, p in enumerate(di_samples)}

train_di = Subset(di_dataset, [di_name_to_index[f] for f in train_filenames if f in di_name_to_index])
val_di   = Subset(di_dataset, [di_name_to_index[f] for f in val_filenames if f in di_name_to_index])
test_di_indices = [di_name_to_index[f] for f in test_filenames if f in di_name_to_index]

di_dataset.transform = test_transform

# Load SD dataset
sd_dataset = ImageFolder(root=sd_path, transform=transform)
sd_samples = sd_dataset.samples
sd_name_to_index = {os.path.basename(p[0]): i for i, p in enumerate(sd_samples)}

train_sd = Subset(sd_dataset, [sd_name_to_index[f] for f in train_filenames if f in sd_name_to_index])
val_sd   = Subset(sd_dataset, [sd_name_to_index[f] for f in val_filenames if f in sd_name_to_index])
test_sd_indices = [sd_name_to_index[f] for f in test_filenames if f in sd_name_to_index]

sd_dataset.transform = test_transform
test_labels = [sd_samples[i][1] for i in test_sd_indices]
class_to_idx = sd_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Custom dataset to test DI/SD models on real test images
class RealTestDataset(Dataset):
    def __init__(self, root, filenames, labels, transform=None):
        self.root = root
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = self.labels[idx]
        class_name = idx_to_class[label]
        path = os.path.join(self.root, class_name, fname)
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

test_di = RealTestDataset(di_path, test_filenames, test_labels, transform=test_transform)
test_sd = RealTestDataset(sd_path, test_filenames, test_labels, transform=test_transform)

# DataLoaders
batch_size = 32
train_loader_real = DataLoader(train_real, batch_size=batch_size, shuffle=True)
val_loader_real   = DataLoader(val_real, batch_size=batch_size, shuffle=False)
test_loader_real  = DataLoader(test_real, batch_size=batch_size, shuffle=False)

train_loader_di = DataLoader(train_di, batch_size=batch_size, shuffle=True)
val_loader_di   = DataLoader(val_di, batch_size=batch_size, shuffle=False)
test_loader_di  = DataLoader(test_di, batch_size=batch_size, shuffle=False)

train_loader_sd = DataLoader(train_sd, batch_size=batch_size, shuffle=True)
val_loader_sd   = DataLoader(val_sd, batch_size=batch_size, shuffle=False)
test_loader_sd  = DataLoader(test_sd, batch_size=batch_size, shuffle=False)

# Model definition
def create_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.apply(lambda m: nn.init.kaiming_normal_(m.weight) if isinstance(m, (nn.Conv2d, nn.Linear)) else None)
    return model.to(device)

# Training + Evaluation loop
def train_and_evaluate(model, train_loader, val_loader, test_loader, label):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_val_acc = 0.0
    for epoch in range(1, 101):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        train_acc = 100 * correct / total

        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        val_acc = 100 * correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_{label}.pth")

        scheduler.step()
        print(f"[{label}] Epoch {epoch}, Loss: {loss_sum:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    model.load_state_dict(torch.load(f"best_model_{label}.pth"))
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    print(f"[{label}] Final Test Accuracy: {100 * correct / total:.2f}%")


# Train and evaluate
print("Training on Real")
model_real = create_model()
train_and_evaluate(model_real, train_loader_real, val_loader_real, test_loader_real, label="real")

print("Training on Diffusion Inversion (DI)")
model_di = create_model()
train_and_evaluate(model_di, train_loader_di, val_loader_di, test_loader_di, label="di")

print("Training on Stable Diffusion (SD)")
model_sd = create_model()
train_and_evaluate(model_sd, train_loader_sd, val_loader_sd, test_loader_sd, label="sd")

