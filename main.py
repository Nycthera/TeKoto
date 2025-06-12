import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def main():
    # ==== Data Transforms ====
    transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ==== Load Full Dataset (no transform yet) ====
    data_root = '/Users/tkrobot/python/asl_to_text/ASL Alphabet Archive/asl_alphabet_train/asl_alphabet_train'
    full_data = datasets.ImageFolder(root=data_root)
    print("Classes:", full_data.classes)

    # ==== Stratified split ====
    indices = list(range(len(full_data)))
    targets = [full_data[i][1] for i in indices]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.167, random_state=42)
    train_idx, val_idx = next(sss.split(indices, targets))

    # Limit train and val sizes if dataset is large
    train_idx = train_idx[:10000]
    val_idx = val_idx[:2000]

    # ==== Create Subsets and assign transforms ====
    # Important: Create full dataset instances with transforms, then Subset
    train_dataset = datasets.ImageFolder(root=data_root, transform=transform_train)
    val_dataset = datasets.ImageFolder(root=data_root, transform=transform_val)

    train_data = Subset(train_dataset, train_idx)
    val_data = Subset(val_dataset, val_idx)

    # Use 0 for num_workers if you encounter issues on some platforms (Windows/macOS)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4)

    # ==== Model: pretrained ResNet18 ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(full_data.classes))
    model = model.to(device)
    torch.save(model.state_dict(), 'asl_model.pth')
    # ==== Loss and Optimizer ====
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ==== Training Loop ====
    num_epochs = 20
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
        scheduler.step()

        print(f"Epoch {epoch + 1} — Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        torch.save(model.state_dict(), 'asl_model_trained.pth')

    # ==== Confusion Matrix & Classification Report ====
    cm = confusion_matrix(all_labels, all_preds)
    class_names = full_data.classes

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == '__main__':
    main()
