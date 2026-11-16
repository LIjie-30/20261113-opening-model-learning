import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from medmnist import PathMNIST
from models.resnet import ResNet18

# 旧 transform（和训练旧权重时完全一致）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True).expand(3, -1, -1)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

root = r"E:\Project\medical_image_learning\project1\data\medmnist"
test_dataset = PathMNIST(split="test", download=False, transform=transform, root=root)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18(num_classes=9).to(device)
model.load_state_dict(torch.load("checkpoints/ResNet18_old.pth", map_location=device))

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.squeeze().long()
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"ResNet18_old Test Accuracy: {100 * correct / total:.2f}%")