import torch
from models.vgg import VGG16
from models.resnet import ResNet18
from models.vit import ViT
from models.vit_handmade import ViTHandmade
from data.dataset import get_data_loaders


def evaluate_model(model, test_loader, device, model_name):
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
    acc = 100 * correct / total
    print(f"{model_name} Test Accuracy: {acc:.2f}%")
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, num_classes = get_data_loaders()

    models = {
        "VGG16": VGG16(num_classes),
        "ResNet18": ResNet18(num_classes),
        "ViT": ViT(num_classes),
    }
    for name, model in models.items():
        model = model.to(device)
        model.load_state_dict(torch.load(f"checkpoints/{name}.pth", map_location=device))
        evaluate_model(model, test_loader, device, name)

    vit_handmade = ViTHandmade(
        image_size=28, patch_size=4, in_channels=3,
        num_classes=9, embed_dim=128, depth=6, num_heads=8
    ).to(device)
    vit_handmade.load_state_dict(torch.load("checkpoints/ViTHandmade.pth", map_location=device))
    evaluate_model(vit_handmade, test_loader, device, "ViTHandmade")


if __name__ == "__main__":
    main()