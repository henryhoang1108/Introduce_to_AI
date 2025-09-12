import torch
import argparse

def test_model(model, dataloader, criterion, device, class_names):
        model_path = "best_model_resnet34_1.pth"
        try:
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        except FileNotFoundError:
            print(f"Error: Model file {model_path} not found.")
            return None, None, None

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        correct_per_class = torch.zeros(len(class_names), dtype=torch.float32, device=device)
        total_per_class = torch.zeros(len(class_names), dtype=torch.float32, device=device)

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for i in range(len(class_names)):
                    class_mask = (labels == i)
                    correct_per_class[i] += (predicted[class_mask] == labels[class_mask]).sum().item()
                    total_per_class[i] += class_mask.sum().item()

        avg_loss = test_loss / total
        accuracy = correct / total
        class_accuracies = correct_per_class / total_per_class
        class_accuracies = class_accuracies.cpu().numpy()

        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print("\nAccuracy per class:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: {class_accuracies[i]:.4f} ({correct_per_class[i]:.0f}/{total_per_class[i]:.0f})")

        return avg_loss, accuracy, class_accuracies

if __name__ == '__main__':
        from model import initialize_model
        from dataloader import get_dataloaders
        import torch.nn as nn

        dataset_dir = "/home/henryhoang/Project/WeatherClassification/data"
        batch_size = 32
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        parser = argparse.ArgumentParser(description='Hyper-parameters for network')
        parser.add_argument('--id', help='Set the model id', default=18, type=int)
        args = parser.parse_args()
        id = args.id  

        dataloaders, total_size, class_names = get_dataloaders(dataset_dir, batch_size)
        test_loader = dataloaders['test']  
        model = initialize_model(num_classes=len(class_names), device=device, id=id)
        criterion = nn.CrossEntropyLoss()

        test_model(model, test_loader, criterion, device, class_names)