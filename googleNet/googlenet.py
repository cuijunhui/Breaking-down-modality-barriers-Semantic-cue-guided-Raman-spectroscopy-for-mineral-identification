import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ---------------------- Inception 模块定义 ----------------------
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, red_3x3, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, red_5x5, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ELU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_proj, kernel_size=1),
            nn.ELU()
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

# ---------------------- GoogLeNet 主干网络 ----------------------
class GoogLeNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.inception1 = InceptionModule(64, 64, 96, 128, 16, 32, 32)    # output: 256
        self.inception2 = InceptionModule(256, 128, 128, 192, 32, 96, 64) # output: 480
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.inception3 = InceptionModule(480, 192, 96, 208, 16, 48, 64)  # output: 512
        self.inception4 = InceptionModule(512, 160, 112, 224, 24, 64, 64) # output: 512

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.global_pool(x)  # [B, 512, 1]
        x = self.classifier(x)   # [B, num_classes]
        return x


# ---------------------- 数据增强函数 ----------------------
def augment_raman_shift(spectra, wavenumber_range=(100, 1799), shift_range=(-0.5, 0.5), num_points=1700):
    start_wavenumber, end_wavenumber = wavenumber_range
    original_wavenumbers = np.linspace(start_wavenumber, end_wavenumber, num_points)
    shift = np.random.uniform(shift_range[0], shift_range[1])
    shifted_wavenumbers = original_wavenumbers + shift

    augmented_spectra = []
    for spectrum in spectra:
        interpolator = interp1d(original_wavenumbers, spectrum, kind='linear', fill_value="extrapolate")
        augmented_spectrum = interpolator(shifted_wavenumbers)
        augmented_spectra.append(augmented_spectrum)

    return np.array(augmented_spectra)


def balance_dataset(spectra, labels, desc_list, wavenumber_range=(100, 1799), shift_range=(-0.5, 0.5)):
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    print(f"Max sample count: {max_count}")

    augmented_spectra = []
    augmented_labels = []
    augmented_desc = []

    for label in unique_labels:
        mask = labels == label
        class_spectra = spectra[mask]
        class_desc = np.array(desc_list)[mask]
        class_count = len(class_spectra)

        if class_count < max_count:
            num_to_augment = max_count - class_count
            indices = np.random.choice(class_count, num_to_augment, replace=True)
            spectra_to_augment = class_spectra[indices]
            desc_to_augment = class_desc[indices]
            augmented = augment_raman_shift(spectra_to_augment, wavenumber_range, shift_range)
            augmented_spectra.append(class_spectra)
            augmented_spectra.append(augmented)
            augmented_labels.extend([label] * class_count)
            augmented_labels.extend([label] * num_to_augment)
            augmented_desc.extend(class_desc.tolist())
            augmented_desc.extend(desc_to_augment.tolist())
        else:
            augmented_spectra.append(class_spectra)
            augmented_labels.extend([label] * class_count)
            augmented_desc.extend(class_desc.tolist())

    augmented_spectra = np.concatenate(augmented_spectra, axis=0)
    augmented_labels = np.array(augmented_labels)
    augmented_desc = augmented_desc
    return augmented_spectra, augmented_labels, augmented_desc

# ---------------------- 数据加载与预处理 ----------------------
file_path = "../excellent_unoriented_data_top20_moveline.csv"
data = pd.read_csv(file_path)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data["Name"].values)
spectra = data.iloc[:, 1:].values
names = data["Name"].values
scaler = StandardScaler()
spectra = scaler.fit_transform(spectra)
desc_list = ["Placeholder description" for _ in names]

# 数据增强
spectra, labels, desc_list = balance_dataset(spectra, labels, desc_list)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    spectra, labels, test_size=0.3, random_state=42, stratify=labels
)

# 检查类别分布
print(pd.Series(labels).value_counts())
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
# 转换为张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# ---------------------- 模型初始化 ----------------------
num_classes = len(label_encoder.classes_)
model = GoogLeNet(input_dim=X_train.shape[1], num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
# optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

# ---------------------- 训练参数 ----------------------
epochs = 100
batch_size = 32
max_grad_norm = 5.0

train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

# ---------------------- 训练循环 ----------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for i in range(0, len(X_train_tensor), batch_size):
        x_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    train_losses.append(total_loss / len(X_train_tensor))
    train_accuracies.append(acc)

    # 测试阶段
    model.eval()
    test_loss = 0
    test_preds, test_labels = [], []

    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            x_batch = X_test_tensor[i:i + batch_size]
            y_batch = y_test_tensor[i:i + batch_size]
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(y_batch.cpu().numpy())

    # 计算指标
    test_acc = accuracy_score(test_labels, test_preds)
    test_losses.append(test_loss / len(X_test_tensor))
    test_accuracies.append(test_acc)

    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']

    # 修改打印输出
    print(f"Epoch [{epoch + 1}/{epochs}] "
          f"Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, "
          f"Train Acc: {acc:.4f}, Test Acc: {test_acc:.4f}, "
          f"LR: {current_lr:.6f}")

    # 如果使用了 scheduler，可以取消注释这行
    # scheduler.step()

# ---------------------- 计算准确率、灵敏度、特异性 ----------------------
def compute_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = []  # Recall
    specificity = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        # Sensitivity (Recall) = TP / (TP + FN)
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivity.append(sens)

        # Specificity = TN / (TN + FP)
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity.append(spec)

    # Macro-average
    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy, avg_sensitivity, avg_specificity, sensitivity, specificity


# 计算测试集的指标
test_accuracy, test_sensitivity, test_specificity, per_class_sensitivity, per_class_specificity = compute_metrics(
    test_labels, test_preds, num_classes
)

# 输出结果
print("\nFinal Test Metrics:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Average Sensitivity (Recall): {test_sensitivity:.4f}")
print(f"Average Specificity: {test_specificity:.4f}")
print("\nPer-class Sensitivity (Recall):")
for i, sens in enumerate(per_class_sensitivity):
    print(f"Class {label_encoder.classes_[i]}: {sens:.4f}")
print("\nPer-class Specificity:")
for i, spec in enumerate(per_class_specificity):
    print(f"Class {label_encoder.classes_[i]}: {spec:.4f}")

# 绘制混淆矩阵
class_names = label_encoder.classes_
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix on Test Set")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("confusion_matrix_top20_google_augmented.png")
plt.show()