import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from tqdm import tqdm
import torch.nn.functional as F

# ====== Config ======
data_dir = "extracted_letters"
image_size = 112
num_classes = 37
batch_size = 64
epochs = 10
learning_rate = 1e-4
val_ratio = 0.1
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

# ====== Transforms ======
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ====== Dataset & Split ======
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# ====== Modified ResNet18 for 1-channel input ======
class GrayResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = GrayResNet18(num_classes).to(device)

# ====== Loss & Optimizer ======
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def evaluate(model, dataloader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    # Training
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total * 100

    # Validation
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
    print(f"   🔹 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   🔸 Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%\n")

# ====== Save for PC ======
torch.save(model.state_dict(), "resnet18_gray_anpr.pth")
print("✅ Saved .pth model for PC")

# ====== Export TorchScript for Pi ======
model.eval()
example_input = torch.randn(1, 1, image_size, image_size)
scripted_model = torch.jit.trace(model.cpu(), example_input)
scripted_model.save("resnet18_gray_anpr_pi.pt")
print("✅ Saved TorchScript model for Raspberry Pi")
