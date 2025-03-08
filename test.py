import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet50, resnet18
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
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
device = "cuda"
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)



class ResNet18SVD(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=False, num_classes=10)

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

            # 截断奇异值分解
            U_trunc = U[:, :r]
            S_trunc = S[:r]
            V_trunc = V[:, :r]

            # 构建分解后的卷积层
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
            # else:
            #     setattr(self.resnet, name, self.replace_linear_with_svd(module))

    def forward(self, x):
        return self.resnet(x)

class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=False, num_classes=10)

    def forward(self, x):
        return self.resnet(x)

class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        return self.resnet(x)

teacher_model = TeacherModel().to(device)
student_model = StudentModel().to(device)
model = ResNet18SVD().to(device)

criterion = nn.CrossEntropyLoss()
optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=0.001)
optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for inputs, labels in progress_bar:
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
            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
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


print("Training Teacher Model (ResNet-50)...")
train_losses_teacher, test_accuracies_teacher = train_model(teacher_model, train_loader, test_loader, criterion, optimizer_teacher, epochs=100)


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
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for inputs, labels in progress_bar:
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
            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        test_accuracy = evaluate_model(student_model, test_loader, criterion)
        train_losses.append(total_loss / len(train_loader))
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")

    return train_losses, test_accuracies


print("Training Student Model (ResNet-18) with Distillation...")
train_losses_distill, test_accuracies_distill = train_distillation(teacher_model, student_model, train_loader, test_loader, optimizer_student, epochs=100)

print("Training Original ResNet-18...")
total_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Total number of parameters: {total_params}")

train_losses_original, test_accuracies_original = train_model(model, train_loader, test_loader, criterion, optimizer,
                                                              epochs=100)
import copy

print("Applying SVD to ResNet-18 with rank=8...")
model_svd_8 = copy.deepcopy(model)
model_svd_8.apply_svd(8)
model_svd_8 = model_svd_8.to(device)
total_params = sum(p.numel() for p in model_svd_8.parameters())
print(model_svd_8)
print(f"Total number of parameters: {total_params}")
optimizer_svd_8 = optim.Adam(model_svd_8.parameters(), lr=0.001)
print("Training SVD-ResNet-18 with rank=8...")
train_losses_svd_8, test_accuracies_svd_8 = train_model(model_svd_8, train_loader, test_loader, criterion, optimizer_svd_8, epochs=100)


print("Applying SVD to ResNet-18 with rank=16...")
model_svd_16 = copy.deepcopy(model)
model_svd_16.apply_svd(16)
model_svd_16 = model_svd_16.to(device)
optimizer_svd_16 = optim.Adam(model_svd_16.parameters(), lr=0.001)
total_params = sum(p.numel() for p in model_svd_16.parameters())
print(model_svd_16)
print(f"Total number of parameters: {total_params}")
print("Training SVD-ResNet-18 with rank=16...")
train_losses_svd_16, test_accuracies_svd_16 = train_model(model_svd_16, train_loader, test_loader, criterion, optimizer_svd_16, epochs=100)

print("Applying SVD to ResNet-18 with rank=32...")
model_svd_32 = copy.deepcopy(model)
model_svd_32.apply_svd(32)
model_svd_32 = model_svd_32.to(device)
optimizer_svd_32 = optim.Adam(model_svd_32.parameters(), lr=0.001)
total_params = sum(p.numel() for p in model_svd_16.parameters())
print(model_svd_32)
print(f"Total number of parameters: {total_params}")
print("Training SVD-ResNet-18 with rank=32...")
train_losses_svd_32, test_accuracies_svd_32 = train_model(model_svd_32, train_loader, test_loader, criterion, optimizer_svd_32, epochs=100)

print("Applying SVD to ResNet-18 with rank=64...")
model_svd_64 = copy.deepcopy(model)
model_svd_64.apply_svd(64)
model_svd_64 = model_svd_64.to(device)
optimizer_svd_64 = optim.Adam(model_svd_64.parameters(), lr=0.001)
total_params = sum(p.numel() for p in model_svd_64.parameters())
print(model_svd_64)
print(f"Total number of parameters: {total_params}")
print("Training SVD-ResNet-18 with rank=64...")
train_losses_svd_64, test_accuracies_svd_64 = train_model(model_svd_64, train_loader, test_loader, criterion, optimizer_svd_64, epochs=100)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses_original, label='Original ResNet-18')
# plt.plot(train_losses_distill, label='Student Model (ResNet-18) with Distillation')
plt.plot(train_losses_svd_8, label='SVD-ResNet-18 (rank=8)')
plt.plot(train_losses_svd_16, label='SVD-ResNet-18 (rank=16)')
plt.plot(train_losses_svd_32, label='SVD-ResNet-18 (rank=32)')
plt.plot(train_losses_svd_64, label='SVD-ResNet-18 (rank=64)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies_original, label='Original ResNet-18')
plt.plot(test_accuracies_distill, label='Student Model (ResNet-18) with Distillation')
plt.plot(test_accuracies_svd_8, label='SVD-ResNet-18 (rank=8)')
plt.plot(test_accuracies_svd_16, label='SVD-ResNet-18 (rank=16)')
plt.plot(test_accuracies_svd_32, label='SVD-ResNet-18 (rank=32)')
plt.plot(test_accuracies_svd_64, label='SVD-ResNet-18 (rank=64)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
