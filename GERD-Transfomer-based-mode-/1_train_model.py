import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

# 从 data 目录加载示例数据集
data_path = os.path.join("data", "sampled_training_data.txt")
final_training_set_mdd = pd.read_csv(data_path, sep="\t")

# 选择输入特征和标签（确保列名与 sample 文件中的一致）
features_mdd = final_training_set_mdd[['chr', 'bpos', '2018_mdd_beta', '2018_mdd_se', '2018_mdd_pval',
                                       '2018_mdd_n', 'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL',
                                       'OCRs_brain', 'OCRs_adult', 'footprints', 'encode',
                                       'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups',
                                       'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']]
labels_mdd = final_training_set_mdd['label']

# 数据标准化
scaler_mdd = StandardScaler()
features_scaled_mdd = scaler_mdd.fit_transform(features_mdd)

# 转换为 Tensor
X_tensor_mdd = torch.tensor(features_scaled_mdd, dtype=torch.float32)
y_tensor_mdd = torch.tensor(labels_mdd.values, dtype=torch.float32).view(-1, 1)

# 使用 DataLoader 构造数据集
dataset_mdd = TensorDataset(X_tensor_mdd, y_tensor_mdd)
train_size_mdd = int(0.8 * len(dataset_mdd))
val_size_mdd = len(dataset_mdd) - train_size_mdd
train_dataset_mdd, val_dataset_mdd = random_split(dataset_mdd, [train_size_mdd, val_size_mdd])
train_loader_mdd = DataLoader(train_dataset_mdd, batch_size=32, shuffle=True)
val_loader_mdd = DataLoader(val_dataset_mdd, batch_size=32, shuffle=False)

# 构建 Transformer 结构的模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim=1):
        super(TransformerClassifier, self).__init__()
        # 将输入映射到一个适合 Transformer 的维度（这里固定为 20）
        self.input_mapping = nn.Linear(input_dim, 20)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=20, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.fc = nn.Linear(20, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_mapping(x)
        # TransformerEncoder 要求输入形状为 (seq_len, batch_size, input_dim)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[0, :, :]  # 取第一个 token 的输出作为分类依据
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# 创建模型实例
input_dim_mdd = features_scaled_mdd.shape[1]
model = TransformerClassifier(input_dim=input_dim_mdd, num_heads=4, num_layers=2, hidden_dim=64)

# 使用 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader_mdd:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        # 为匹配 Transformer 输入，增加一个序列维度（1）
        outputs = model(X_batch.unsqueeze(1))
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 在验证集上评估
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_val, y_val in val_loader_mdd:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val.unsqueeze(1))
            loss = criterion(outputs, y_val)
            val_loss += loss.item()
            predicted = (outputs > 0.9).float()  # 采用 0.9 作为分类阈值
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader_mdd):.4f}, "
          f"Val Loss: {val_loss/len(val_loader_mdd):.4f}, Accuracy: {accuracy:.2f}%")

# 最终在验证集上评估模型表现
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_val, y_val in val_loader_mdd:
        X_val, y_val = X_val.to(device), y_val.to(device)
        outputs = model(X_val.unsqueeze(1))
        predicted = (outputs > 0.9).float()
        total += y_val.size(0)
        correct += (predicted == y_val).sum().item()
final_accuracy = 100 * correct / total
print(f"Final Accuracy on Validation Set: {final_accuracy:.2f}%")

# 保存训练后的模型到 model 目录
model_save_path = os.path.join("model", "transformer_model_mdd.pth")
os.makedirs("model", exist_ok=True)  # 如果 model 目录不存在则创建
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存到 {model_save_path}")