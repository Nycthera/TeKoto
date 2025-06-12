import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm


def main():
    # ==== Transform ====
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Smaller size for speed
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ==== Load Data ====
    data = datasets.ImageFolder(
        root='/Users/tkrobot/python/asl_to_text/ASL Alphabet Archive/asl_alphabet_train/asl_alphabet_train',
        transform=transform
    )

    print("Classes:", data.classes)

    # ==== Smaller Subsets ====
    indices = list(range(len(data)))
    train_indices, val_indices = train_test_split(indices, train_size=0.9, random_state=42)

    random.seed(42)
    random.shuffle(train_indices)
    random.shuffle(val_indices)

    small_train_indices = train_indices[:1500]
    small_val_indices = val_indices[:1000]

    train_data = Subset(data, small_train_indices)
    val_data = Subset(data, small_val_indices)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=64, num_workers=2)

    # ==== Model ====
    class ASLCNN(nn.Module):
        def __init__(self):
            super(ASLCNN, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),  # [B, 32, 112, 112]
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),                # [B, 32, 56, 56]

                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),                # [B, 64, 28, 28]

                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),                # [B, 128, 14, 14]

                nn.Dropout(0.3),
            )

            self.fc = nn.Sequential(
                nn.Linear(128 * 14 * 14, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(data.classes))  # Output layer for classification
            )

        def forward(self, x):
            x = self.net(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASLCNN().to(device)

    # Optional: use torch.compile if available for speed (PyTorch 2.0+)
    # Uncomment the following line if your PyTorch supports it:
    # model = torch.compile(model)

    # Weight initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # ==== Train Loop ====
    num_epochs = 10
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ==== Validation ====
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / total
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch + 1} — Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # ==== Final Evaluation: Confusion Matrix ====
    cm = confusion_matrix(all_labels, all_preds)
    class_names = data.classes

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # ==== Optional: Classification Report ====
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == '__main__':
    main()
