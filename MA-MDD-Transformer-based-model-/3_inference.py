import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
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
        x = x.permute(1, 0, 2)  # Convert dimensions to (seq_len, batch_size, input_dim)
        x = self.transformer(x)
        x = x[0, :, :]  # Select the output of the first token
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
# Load the fine-tuned model for inference
# Load inference data
inference_data_path = 'data/Inference_Data_MA.txt'
data = pd.read_csv(inference_data_path, sep='\t')

model = TransformerClassifier(input_dim=22, num_heads=4, num_layers=2, hidden_dim=64)  # Input dimension set to 21
model.load_state_dict(torch.load('model/MA_transform_learning.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()



# Select inference features
inference_features = data[['chr', 'bpos', '2020_beta', '2020_se', '2020_pval', '2020_n',
                           'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL', 
                           'OCRs_brain', 'OCRs_adult', 'footprints', 'encode', 
                           'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups', 
                           'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']]

# Standardize data (assume a fresh scaler as inference data might be scaled independently)
scaler_inference = StandardScaler()
inference_features_scaled = scaler_inference.fit_transform(inference_features)

# Convert to Tensor
X_inference_tensor = torch.tensor(inference_features_scaled, dtype=torch.float32).to(device)

# Perform inference
with torch.no_grad():
    outputs = model(X_inference_tensor.unsqueeze(1))  # Add extra dimension for transformer input
    predictions = (outputs > 0.999).float().cpu().numpy()  # Apply threshold of 0.999 for classification

# Select rows where prediction is 1
predicted_snps = data.loc[predictions.flatten() == 1, ['snp', 'chr', 'bpos', '2020_pval']]
predicted_snps.rename(columns={'2020_pval': 'pval'}, inplace=True)

# Save the results to a text file
output_path = 'Predicted_SNPs.txt'
predicted_snps.to_csv(output_path,sep='\t', index=False)
print(f"Inference results saved to {output_path}")