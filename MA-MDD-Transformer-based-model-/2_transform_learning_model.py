import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
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

data_path = 'data/Train_Data_MA.txt'
final_training_set_data = pd.read_csv(data_path, sep='\t')
features_data = final_training_set_data[['chr', 'bpos', '2016_beta', '2016_se', '2016_pval', '2016_n', 
                                         'sqtl', 'sc-eqtl', 'brain_eQTL', 'all_eQTL', 'mQTL', 
                                         'OCRs_brain', 'OCRs_adult', 'footprints', 'encode',
                                         'targetScanS.wgRna', 'tfbsConsSites', 'genomicSuperDups', 
                                         'reported in previous GWAS', 'ldscore', 'freq', 'ALL_pred']]
labels_data = final_training_set_data['label']

# Standardize features
scaler_data = StandardScaler()
features_scaled_data = scaler_data.fit_transform(features_data)

# Convert to tensors
X_tensor_data = torch.tensor(features_scaled_data, dtype=torch.float32)
y_tensor_data = torch.tensor(labels_data.values, dtype=torch.float32).view(-1, 1)

# Use DataLoader
dataset_data = TensorDataset(X_tensor_data, y_tensor_data)

# Split dataset into training and validation sets
train_size_data = int(0.8 * len(dataset_data))
val_size_data = len(dataset_data) - train_size_data
train_dataset_data, val_dataset_data = random_split(dataset_data, [train_size_data, val_size_data])

train_loader_data = DataLoader(train_dataset_data, batch_size=32, shuffle=True)
val_loader_data = DataLoader(val_dataset_data, batch_size=32, shuffle=False)

# Load pre-trained model
model = TransformerClassifier(input_dim=features_scaled_data.shape[1], num_heads=4, num_layers=2, hidden_dim=64)
model.load_state_dict(torch.load('model/MDD_model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Smaller learning rate for transfer learning

# Train the model with transfer learning
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader_data:
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
        for X_val, y_val in val_loader_data:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val.unsqueeze(1))
            loss = criterion(outputs, y_val)
            val_loss += loss.item()

            predicted = (outputs > 0.9).float()
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader_data)}, "
          f"Val Loss: {val_loss/len(val_loader_data)}, Accuracy: {accuracy}%")

# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_val, y_val in val_loader_data:
        X_val, y_val = X_val.to(device), y_val.to(device)
        outputs = model(X_val.unsqueeze(1))
        predicted = (outputs > 0.9).float()
        total += y_val.size(0)
        correct += (predicted == y_val).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy on Validation Set: {accuracy}%")

# Save the fine-tuned model
model_save_path = 'model/MA_transform_learning.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")