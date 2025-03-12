import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# **1. 设定路径**
DATA_PATH = "data/sampled_inference_data.txt"  # 读取数据
MODEL_DIR = "model"
PRETRAINED_MODEL_PATH = os.path.join(MODEL_DIR, "transformer_model_mdd.pth")
FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, "transformer_model_finetuned.pth")

# **2. 确保 `model/` 目录存在**
os.makedirs(MODEL_DIR, exist_ok=True)

# **3. 读取推理数据**
df = pd.read_csv(DATA_PATH, sep="\t")

# **4. 选择输入特征和标签**
feature_cols = ['chr', 'bpos', '2016_beta', '2016_se', '2016_pval', '2016_n', 
                'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL', 
                'OCRs_brain', 'OCRs_adult', 'footprints', 'encode',
                'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups', 
                'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']
label_col = 'label'

features = df[feature_cols]
labels = df[label_col]

# **5. 数据标准化**
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# **6. 转换为 Tensor**
X_tensor = torch.tensor(features_scaled, dtype=torch.float32)
y_tensor = torch.tensor(labels.values, dtype=torch.float32).view(-1, 1)

# **7. 使用 DataLoader**
dataset = TensorDataset(X_tensor, y_tensor)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# **8. 构建 Transformer 结构的模型**
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim=1):
        super(TransformerClassifier, self).__init__()
        self.input_mapping = nn.Linear(input_dim, 20)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=20, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.fc = nn.Linear(20, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_mapping(x)
        x = x.permute(1, 0, 2)  # 变换维度 (seq_len, batch_size, input_dim)
        x = self.transformer(x)
        x = x[0, :, :]
        x = self.fc(x)
        return self.sigmoid(x)

# **9. 加载预训练模型**
input_dim = features_scaled.shape[1]
model = TransformerClassifier(input_dim=input_dim, num_heads=4, num_layers=2, hidden_dim=64)

# 检查模型是否存在
if os.path.exists(PRETRAINED_MODEL_PATH):
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    print(f"成功加载预训练模型: {PRETRAINED_MODEL_PATH}")
else:
    print(f"预训练模型未找到: {PRETRAINED_MODEL_PATH}")
    exit()

# **10. 设备配置**
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# **11. 定义损失函数和优化器**
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 迁移学习使用较小的学习率

# **12. 迁移学习训练模型**
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(1))  # 添加一维以匹配 Transformer 输入
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # **13. 计算验证集损失**
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val.unsqueeze(1))
            loss = criterion(outputs, y_val)
            val_loss += loss.item()

            predicted = (outputs > 0.99).float()
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {accuracy:.2f}%")

# **14. 评估模型**
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_val, y_val in val_loader:
        X_val, y_val = X_val.to(device), y_val.to(device)
        outputs = model(X_val.unsqueeze(1))
        predicted = (outputs > 0.99).float()
        total += y_val.size(0)
        correct += (predicted == y_val).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy on Validation Set: {accuracy:.2f}%")

# **15. 保存迁移学习后的模型**
torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
print(f"✅ 迁移学习后的模型已保存到: {FINETUNED_MODEL_PATH}")
