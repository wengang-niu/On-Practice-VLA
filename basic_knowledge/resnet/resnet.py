"""一个简单的 ResNet18 猫狗二分类器。

使用公开的 Oxford-IIIT Pet 数据集，只训练预训练 ResNet18 的最后一层分类器。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# 1. 基本设置
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_dataloaders():
    """构建训练和验证数据加载器。"""
    train_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="binary-category",
        transform=transform,
        download=True,
    )

    val_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="test",
        target_types="binary-category",
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def build_model():
    """加载预训练 ResNet18 并替换最后一层。"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


def train(model, train_loader, val_loader):
    """训练和评估模型。"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_loss = train_loss / train_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print(
            f"epoch {epoch + 1}/{EPOCHS}, "
            f"loss: {avg_loss:.4f}, "
            f"train acc: {train_acc:.4f}, "
            f"val acc: {val_acc:.4f}"
        )


def main():
    train_loader, val_loader = build_dataloaders()
    model = build_model()
    train(model, train_loader, val_loader)

    torch.save(model.state_dict(), "resnet_cat_dog.pth")
    print("saved model to resnet_cat_dog.pth")


if __name__ == "__main__":
    main()
