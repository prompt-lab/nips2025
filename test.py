import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# from torchvision.models import resnet50, resnet18
from models.resnet import resnet50, resnet18
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import copy

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataset(dataset_name, root='./data', train=True, download=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'caltech101':
        dataset = torchvision.datasets.Caltech101(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'oxford_pets':
        dataset = torchvision.datasets.OxfordIIITPet(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'stanford_cars':
        dataset = torchvision.datasets.StanfordCars(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'oxford_flowers':
        dataset = torchvision.datasets.Flowers102(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'food101':
        dataset = torchvision.datasets.Food101(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'fgvc_aircraft':
        dataset = torchvision.datasets.FGVCAircraft(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'sun397':
        dataset = torchvision.datasets.SUN397(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'dtd':
        dataset = torchvision.datasets.DTD(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'eurosat':
        dataset = torchvision.datasets.EuroSAT(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'ucf101':
        dataset = torchvision.datasets.UCF101(root=root, download=download, transform=transform_train if train else transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset

def get_dataloader(dataset_name, batch_size=128, shuffle=True, root='./data', download=True):
    train_set = get_dataset(dataset_name, root=root, train=True, download=download)
    test_set = get_dataset(dataset_name, root=root, train=False, download=download)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class ResNet18SVD(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = resnet18(num_classes=num_classes)

    def replace_linear_with_svd(self, module, rank):
        if isinstance(module, nn.Linear):
            in_dim, out_dim = module.in_features, module.out_features
            weight = module.weight.data
            bias = module.bias.data.clone() if module.bias is not None else None

            U, S, V = torch.svd(weight)
            if rank > S.numel():
                return module
            r = min(rank, S.numel())

            U_trunc = U[:, :r]
            S_trunc = S[:r]
            V_trunc = V[:, :r]

            B = U_trunc @ torch.diag(S_trunc)
            A = V_trunc.t()

            svd_layer = nn.Sequential(
                nn.Linear(in_dim, r, bias=False),
                nn.Linear(r, out_dim, bias=True)
            )
            svd_layer[0].weight.data = A
            svd_layer[1].weight.data = B
            svd_layer[1].bias.data = bias
            return svd_layer
        return module

    def replace_conv_with_svd(self, module, rank):
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            C_out, C_in, K_h, K_w = weight.shape
            weight_flat = weight.view(C_out, -1)
            U, S, V = torch.svd(weight_flat)
            if rank >= S.numel():
                return module
            r = min(rank, S.numel())

            U_trunc = U[:, :r]
            S_trunc = S[:r]
            V_trunc = V[:, :r]

            conv1 = nn.Conv2d(C_in, r, kernel_size=(K_h, K_w), stride=module.stride, padding=module.padding, bias=False)
            conv2 = nn.Conv2d(r, C_out, kernel_size=1, stride=1, padding=0, bias=True)

            conv1.weight.data = V_trunc.t().view(r, C_in, K_h, K_w)
            conv2.weight.data = (U_trunc @ torch.diag(S_trunc)).view(C_out, r, 1, 1)
            try:
                conv2.bias.data = module.bias.data.clone() if module.bias is not None else None
            except:
                pass
            return nn.Sequential(conv1, conv2)
        elif isinstance(module, nn.Sequential) and len(module) == 2 and isinstance(module[0], nn.Conv2d):
            conv = module[0]
            bn = module[1]
            svd_conv = self.replace_conv_with_svd(conv, rank)
            return nn.Sequential(svd_conv, bn)
        return module

    def apply_svd(self, rank):
        for name, module in self.resnet.named_children():
            if isinstance(module, nn.Sequential):
                for block_name, block in module.named_children():
                    for layer_name, layer in block.named_children():
                        setattr(block, layer_name, self.replace_conv_with_svd(layer, rank))

    def initialize_weights_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def initialize_weights_gaussian(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        return self.resnet(x)

class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = resnet50(num_classes=num_classes)

    def forward(self, x):
        return self.resnet(x)

class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.resnet(x)

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        test_accuracy = evaluate_model(model, test_loader, criterion)
        train_losses.append(total_loss / len(train_loader))
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
    return train_losses, test_accuracies

def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_distillation(teacher_model, student_model, train_loader, test_loader, optimizer, temp=7, alpha=0.3, epochs=10):
    student_model.train()
    teacher_model.eval()
    train_losses = []
    test_accuracies = []
    hard_loss = nn.CrossEntropyLoss()
    soft_loss = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)
            student_loss = hard_loss(student_outputs, labels)
            distillation_loss = soft_loss(
                F.log_softmax(student_outputs / temp, dim=1),
                F.softmax(teacher_outputs / temp, dim=1)
            )
            loss = alpha * student_loss + (1 - alpha) * temp * temp * distillation_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        test_accuracy = evaluate_model(student_model, test_loader, criterion)
        train_losses.append(total_loss / len(train_loader))
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
    return train_losses, test_accuracies

def train_svd_model(model, rank, train_loader, test_loader, criterion, optimizer, init_method=None, epochs=10, teacher_model=None):
    model_svd = copy.deepcopy(model)
    model_svd.apply_svd(rank)
    if init_method == "distillation":
        model_svd = model_svd.to(device)
        optimizer_student = optim.Adam(model_svd.parameters(), lr=0.001)
        train_losses, test_accuracies = train_distillation(teacher_model, model_svd, train_loader, test_loader, optimizer_student, epochs=100)
        return train_losses, test_accuracies
    if init_method == 'kaiming':
        model_svd.initialize_weights_kaiming()
    elif init_method == 'gaussian':
        model_svd.initialize_weights_gaussian()
    model_svd = model_svd.to(device)
    optimizer_svd = optim.Adam(model_svd.parameters(), lr=0.001)
    train_losses, test_accuracies = train_model(model_svd, train_loader, test_loader, criterion, optimizer_svd, epochs=epochs)
    return train_losses, test_accuracies

dataset_name = 'cifar100'
train_loader, test_loader = get_dataloader(dataset_name)

num_classes = 100
teacher_model = TeacherModel(num_classes=num_classes).to(device)
student_model = StudentModel(num_classes=num_classes).to(device)
model = ResNet18SVD(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=0.001)
optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training Teacher Model (ResNet-50)...")
train_losses_teacher, test_accuracies_teacher = train_model(teacher_model, train_loader, test_loader, criterion, optimizer_teacher, epochs=100)

print("Training Student Model (ResNet-18) with Distillation...")
train_losses_distill, test_accuracies_distill = train_distillation(teacher_model, student_model, train_loader, test_loader, optimizer_student, epochs=100)

print("Training Original ResNet-18...")
train_losses_original, test_accuracies_original = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=100)

ranks = [16, 32, 64]
init_methods = [None, 'kaiming', 'distillation']
results = {}

for rank in ranks:
    for init_method in init_methods:
        key = f"rank_{rank}_{init_method if init_method else 'default'}"
        print(f"Training SVD-ResNet-18 with {key}...")
        train_losses, test_accuracies = train_svd_model(model, rank, train_loader, test_loader, criterion, optimizer, init_method=init_method, epochs=100, teacher_model=teacher_model)
        results[key] = (train_losses, test_accuracies)

plt.figure(figsize=(12,8))

plt.subplot(1, 2, 1)
plt.plot(train_losses_original, label='Original ResNet-18')
for key, (train_losses, _) in results.items():
    plt.plot(train_losses, label=f'SVD-ResNet-18 ({key})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies_original, label='Original ResNet-18')
plt.plot(test_accuracies_distill, label='Student Model (ResNet-18) with Distillation')
for key, (_, test_accuracies) in results.items():
    plt.plot(test_accuracies, label=f'SVD-ResNet-18 ({key})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("result.png")
plt.show()
