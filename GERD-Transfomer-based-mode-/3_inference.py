import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# **1. 设置路径**
DATA_PATH = "data/sampled_prediction_data.txt"  # 推理数据
MODEL_PATH = "model/transformer_model_finetuned.pth"  # 迁移学习后的模型

# **2. 设备配置**
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# **3. 读取推理数据**
df = pd.read_csv(DATA_PATH, sep="\t")

# **4. 选择推理输入特征**
feature_cols = ['chr', 'bpos', '2016_beta', '2016_se', '2016_pval', '2016_n', 
                'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL', 
                'OCRs_brain', 'OCRs_adult', 'footprints', 'encode', 
                'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups', 
                'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']

# **5. 提取输入特征**
inference_features = df[feature_cols]

# **6. 数据标准化（使用新 StandardScaler 适配当前推理数据）**
scaler = StandardScaler()
inference_features_scaled = scaler.fit_transform(inference_features)  # 仅基于当前推理数据进行标准化

# **7. 转换为 Tensor**
X_inference_tensor = torch.tensor(inference_features_scaled, dtype=torch.float32).to(device)

# **8. 构建 Transformer 结构的模型**
class TransformerClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim=1):
        super(TransformerClassifier, self).__init__()
        self.input_mapping = torch.nn.Linear(input_dim, 20)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=20, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.fc = torch.nn.Linear(20, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.input_mapping(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
        x = self.transformer(x)
        x = x[0, :, :]
        return self.sigmoid(self.fc(x))

# **9. 加载模型**
input_dim = inference_features_scaled.shape[1]
model = TransformerClassifier(input_dim=input_dim, num_heads=4, num_layers=2, hidden_dim=64)

# 检查模型是否存在
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"成功加载迁移学习后的模型: {MODEL_PATH}")
else:
    print(f"迁移学习后的模型未找到: {MODEL_PATH}")
    exit()

model.to(device)
model.eval()

# **10. 进行推理**
with torch.no_grad():
    outputs = model(X_inference_tensor.unsqueeze(1))
    predictions = (outputs > 0.999).float().cpu().numpy()

# **11. 筛选推理为 1 的行**
predicted_snps = df.loc[predictions.flatten() == 1, 'snp']


# **13. 保存推理结果**
OUTPUT_PATH = "data/predicted_snps.txt"
predicted_snps.to_csv(OUTPUT_PATH, sep="\t", index=False)

print(f"推理结果已保存至: {OUTPUT_PATH}")