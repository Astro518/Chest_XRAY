import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import os
import numpy as np

# Speed optimization
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'C:\\Users\\Ayush\\.vscode\\Projects\\dl project\\chest_xray'


# Transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_test)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_test)

# Compute class weights
class_counts = [0, 0]
for _, label in train_dataset.samples:
    class_counts[label] += 1
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum()

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n Test Accuracy: {100 * np.mean(np.array(all_preds) == np.array(all_labels)):.2f}%")
    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    print("\n Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# Training loop
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=3):
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Validation accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# Hyperparameter tuning
if __name__ == '__main__':
    batch_sizes = [16, 32]
    learning_rates = [1e-4, 1e-3]
    weight_decays = [0, 1e-4]
    best_acc = 0.0
    best_params = {}
    best_model_state = None

    for batch_size in batch_sizes:
        for lr in learning_rates:
            for wd in weight_decays:
                print(f"\n Tuning: batch_size={batch_size}, lr={lr}, weight_decay={wd}")

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

                # Load fresh model
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                model.fc = nn.Linear(model.fc.in_features, 2)
                model = model.to(device)

                for param in model.parameters():
                    param.requires_grad = True

                criterion = nn.CrossEntropyLoss(weight=weights.to(device))
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

                val_acc = train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=3)
                print(f" Validation Accuracy: {val_acc:.2f}%")

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = {'batch_size': batch_size, 'lr': lr, 'weight_decay': wd}
                    best_model_state = model.state_dict()

    # Final evaluation
    print(f"\n Best Hyperparameters: {best_params}")
    model.load_state_dict(best_model_state)
    model = model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], num_workers=2)
    evaluate_model(model, test_loader)
