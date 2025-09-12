import time
import copy
import torch
import matplotlib.pyplot as plt
from loss import get_criterion
from dataloader import get_dataloaders
from model import initialize_model
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs, early_stopping):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        train_loss_history = []
        val_loss_history = []
        accuracy_history = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()
                    train_loss_history.append(running_loss / dataset_sizes[phase])

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    val_loss_history.append(epoch_loss)
                    accuracy_history.append(epoch_acc.item())

                    early_stopping(epoch_loss)

                    if early_stopping.early_stop:
                        print("Early stopping!")
                        model.load_state_dict(best_model_wts)
                        return model, train_loss_history, val_loss_history, accuracy_history
    
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(best_model_wts)
        return model, train_loss_history, val_loss_history, accuracy_history

def plot_loss_accuracy(train_loss, val_loss, accuracy):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label="Training Loss", color='red')
        plt.plot(val_loss, label="Validation Loss", color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(accuracy, label="Validation Accuracy", color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'curves_{id}_1.png')
        plt.show()

if __name__ == "__main__":
        dataset_dir = "/home/henryhoang/Project/WeatherClassification/data"
        dataloaders, dataset_sizes, class_names = get_dataloaders(dataset_dir)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        parser = argparse.ArgumentParser(description='Hyper-parameters for network')
        parser.add_argument('--id', help='Set the model id', default=18, type=int)
        args = parser.parse_args()
        id = args.id
        
        model = initialize_model(len(class_names), device, id)
        criterion = get_criterion()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        early_stopping = EarlyStopping(patience=7, min_delta=0.001)
        num_epochs = 50
        
        model, train_loss, val_loss, accuracy = train_model(
            model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs, early_stopping)
        
        torch.save(model.state_dict(), f"best_model_resnet{id}_1.pth")
        plot_loss_accuracy(train_loss, val_loss, accuracy)