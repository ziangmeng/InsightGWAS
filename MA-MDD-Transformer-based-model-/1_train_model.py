import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

data_path = 'data/Train_Data_MDD.txt'
final_training_set_mdd = pd.read_csv(data_path, sep='\t')

features_mdd = final_training_set_mdd[['chr', 'bpos', '2018_mdd_beta', '2018_mdd_se', '2018_mdd_pval',
                                       '2018_mdd_n', 'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL',
                                       'OCRs_brain', 'OCRs_adult', 'footprints', 'encode',
                                       'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups',
                                       'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']]
labels_mdd = final_training_set_mdd['label']

# Standardize features
scaler_mdd = StandardScaler()
features_scaled_mdd = scaler_mdd.fit_transform(features_mdd)

# Convert to tensors
X_tensor_mdd = torch.tensor(features_scaled_mdd, dtype=torch.float32)
y_tensor_mdd = torch.tensor(labels_mdd.values, dtype=torch.float32).view(-1, 1)

# Use DataLoader
dataset_mdd = TensorDataset(X_tensor_mdd, y_tensor_mdd)

# Split dataset into training and validation sets
train_size_mdd = int(0.8 * len(dataset_mdd))
val_size_mdd = len(dataset_mdd) - train_size_mdd
train_dataset_mdd, val_dataset_mdd = random_split(dataset_mdd, [train_size_mdd, val_size_mdd])

train_loader_mdd = DataLoader(train_dataset_mdd, batch_size=32, shuffle=True)
val_loader_mdd = DataLoader(val_dataset_mdd, batch_size=32, shuffle=False)

# Define Transformer-based model
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
        x = x.permute(1, 0, 2)  # Reorder dimensions to (seq_len, batch_size, input_dim)
        x = self.transformer(x)
        x = x[0, :, :]  # Use the first token's output
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Initialize model
input_dim_mdd = features_scaled_mdd.shape[1]
model = TransformerClassifier(input_dim=input_dim_mdd, num_heads=4, num_layers=2, hidden_dim=64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader_mdd:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(1))  # Add an extra dimension for Transformer input
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
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

            predicted = (outputs > 0.9).float()
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader_mdd)}, "
          f"Val Loss: {val_loss/len(val_loader_mdd)}, Accuracy: {accuracy}%")

# Evaluate model
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

accuracy = 100 * correct / total
print(f"Final Accuracy on Validation Set: {accuracy}%")

# Save model
model_save_path = 'model/MDD_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")