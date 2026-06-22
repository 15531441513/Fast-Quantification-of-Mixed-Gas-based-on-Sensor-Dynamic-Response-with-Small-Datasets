import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import h5py
import time
import pandas as pd
import joblib


# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Parameter configuration
No = 0
case = 0
lent=1247
HidNeuron = 20
num_steps = lent*10
batch_size = 20
Epo = 400
neuron = 14
GAS=['PdNi','PANI','Te']

# --------------------- Data loading ---------------------
# Original data path
hdf5_file_path = 'setD0'  + 'conallrandom.h5'

# Read HDF5 data (keep original dimension order)
with h5py.File(hdf5_file_path, 'r') as f:
    # Dimension order: (samples, time_steps, features)
    X_train = torch.tensor(f['X_train'][: , :, [0, 1, 2, 3+ No]], dtype=torch.float32)  # [batch, steps, features]
    Y_train = torch.tensor(f['Y_train'][:, :, No], dtype=torch.float32)  # [batch, steps]

    X_valid = torch.tensor(f['X_valid'][:, :, [0, 1, 2, 3+ No]], dtype=torch.float32)
    Y_valid = torch.tensor(f['Y_valid'][:, :, No], dtype=torch.float32)

    X_test = torch.tensor(f['X_test'][:, :, [0, 1, 2, 3+ No]], dtype=torch.float32)
    Y_test = torch.tensor(f['Y_test'][:, :, No], dtype=torch.float32)

# Create DataLoader (keep original dimensions)
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# for i in range(4):
#     plt.plot(X_valid[1,:, i], label=f'Dimension {i+1}')
#
# # Add legend and labels
# plt.legend()
# plt.title('Line Plots for Each Dimension of the Tensor')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid(True)

# Show plot
plt.show()
# --------------------- Model definition ---------------------
class TimeStepGRU(nn.Module):  # Class renamed to TimeStepGRU
    def __init__(self, input_size, hidden_size):
        super(TimeStepGRU, self).__init__()
        self.gru = nn.GRU(  # Replace LSTM with GRU
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # GRU layer output: (batch, steps, hidden_size)
        output, hn = self.gru(x)  # GRU has no cell state, only returns hidden state
        output = self.dropout(output)
        out = self.fc(output)  # [batch, steps, 1]
        return self.sigmoid(out.squeeze(-1))


# # Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeStepGRU(input_size=4, hidden_size=HidNeuron).to(device)  # Using the new class name
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()
#
# Move data to device
X_train, Y_train = X_train.to(device), Y_train.to(device)
X_valid, Y_valid = X_valid.to(device), Y_valid.to(device)

# Record training progress
train_losses = []
val_losses = []

# --------------------- Training loop ---------------------
start_time = time.time()

for epoch in range(Epo):
    # Training mode
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validation mode
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid)
        val_loss = criterion(val_outputs, Y_valid)

    # Record losses
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss.item())

    # Print training info
    print(f'Epoch [{epoch + 1:03d}/{Epo}] | '
          f'Train Loss: {train_loss:.4f} | '
          f'Val Loss: {val_loss.item():.4f}')

end_time = time.time()
print(f"\nTotal training time: {end_time - start_time:.2f} seconds")

# # --------------------- Save model ---------------------
model_save_path = f'modelD0{case}1.pth'
torch.save(model.state_dict(), model_save_path)

# --------------------- Visualization ---------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, Epo + 1), train_losses, '#A61B24', label='Training Loss')
plt.plot(range(1, Epo + 1), val_losses, '#C04851', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0, Epo + 1)
plt.legend()
plt.title(f'Training and Validation Loss Curves of Sensor {GAS[No]}')
# plt.show()

# --------------------- Fine-tune model ---------------------

# Load trained model
model.load_state_dict(torch.load(model_save_path))

# Freeze GRU layer parameters so they don't update during fine-tuning
for param in model.gru.parameters():
    param.requires_grad = False

# Modify forward pass (dynamically add feature slicing)
def new_forward(self, x):
    output, hn = self.gru(x)
    output = self.dropout(output)
    # Only use the last n hidden neurons
    output = output[:, :, -neuron:]  # Slice to extract last 5 features
    out = self.fc(output)
    return self.sigmoid(out.squeeze(-1))

# Replace fully connected layer (input dimension changed to n)
model.fc = nn.Linear(neuron, 1).to(device)  # Key modification point

# Bind the new forward method
import types
model.forward = types.MethodType(new_forward, model)

# Only optimize the fully connected layer
optimizer_fine = optim.Adam(model.fc.parameters(), lr=0.001)


# Fine-tuning epoch settings
Epo_finetune = Epo  # Adjust fine-tuning epochs as needed

# Record fine-tuning process
fine_train_losses = []
fine_val_losses = []

# Fine-tuning training loop
start_time_fine = time.time()

for epoch in range(Epo_finetune):
    model.train()  # Ensure dropout etc. training mode is enabled
    epoch_loss = 0

    # Training batches
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer_fine.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_fine.step()

        epoch_loss += loss.item()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid)
        val_loss = criterion(val_outputs, Y_valid)

    # Record losses
    train_loss = epoch_loss / len(train_loader)
    fine_train_losses.append(train_loss)
    fine_val_losses.append(val_loss.item())

    # Print info
    print(f'Fine-tune Epoch [{epoch+1:03d}/{Epo_finetune}] | '
          f'Train Loss: {train_loss:.4f} | '
          f'Val Loss: {val_loss.item():.4f}')

end_time_fine = time.time()
print(f"\nTotal fine-tuning time: {end_time_fine - start_time_fine:.2f} seconds")

# Save fine-tuned model
fine_model_path =  f'modelD0{No}1_fine.pth'
torch.save(model.state_dict(), fine_model_path)

# Visualize fine-tuning process
plt.figure(figsize=(10, 6))
plt.plot(range(1, Epo_finetune+1), fine_train_losses, '#2A5CAA', label='Training Loss (Fine)')
plt.plot(range(1, Epo_finetune+1), fine_val_losses, '#1E90FF', label='Validation Loss (Fine)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0, Epo_finetune+1)
plt.legend()
plt.title(f'Fine-tuning Loss Curves of Sensor {GAS[No]}')
plt.show()


# Create DataFrame with index
df = pd.DataFrame(fine_train_losses)


# Save to CSV file
df.to_csv('fine_train_losses.csv')
print(f"Successfully saved relative error data to: {'fine_train_losses.csv'}")

# Create DataFrame with index
df = pd.DataFrame(fine_val_losses)


# Save to CSV file
df.to_csv('fine_val_losses.csv')
print(f"Successfully saved relative error data to: {'fine_val_losses.csv'}")



# --------------------- Must rebuild model structure before prediction ---------------------
class FineTunedGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FineTunedGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(neuron, 1)  # 输入维度固定为5
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass logic consistent with fine-tuning
        output, _ = self.gru(x)
        output = self.dropout(output)
        output = output[:, :, -neuron:]  # Slice last 5 features
        out = self.fc(output)
        return self.sigmoid(out.squeeze(-1))


# --------------------- Correct way to load fine-tuned model ---------------------
def load_finetuned_model(model_path, hidden_size):
    # 1. Initialize model structure
    model = FineTunedGRU(
        input_size=4,
        hidden_size=hidden_size
    ).to(device)

    # 2. Load trained parameters
    model.load_state_dict(torch.load(model_path))

    # 3. Freeze GRU layer parameters (consistent with training)
    for param in model.gru.parameters():
        param.requires_grad = False

    # 4. Set evaluation mode
    model.eval()
    return model


# --------------------- Usage example ---------------------
# 加载基础模型和微调模型
base_model = TimeStepGRU(input_size=4, hidden_size=HidNeuron).to(device)
print(HidNeuron)
base_model.load_state_dict(torch.load(f'modelD0{case}1.pth'))

fine_model = load_finetuned_model(
    model_path=f'modelD0{No}1_fine.pth',
    hidden_size=HidNeuron
)
print(fine_model)


# --------------------- Prediction and visualization ---------------------
def inverse_transform(preds, scaler_path):
    """Inverse transform (denormalize) predictions"""
    scaler = joblib.load(scaler_path)
    return scaler.inverse_transform(preds)

# Perform prediction (example)
with torch.no_grad():
    # Base model prediction
    base_preds = base_model(X_test.to(device)).cpu().numpy()
    # Fine-tuned model prediction (using modified structure)
    fine_preds = fine_model(X_test.to(device)).cpu().numpy()

# Inverse transform (denormalize)
scaler_path =  f'yscalarD0{No}1.joblib'
base_preds = inverse_transform(base_preds.reshape(-1, 1), scaler_path).reshape(base_preds.shape)
fine_preds = inverse_transform(fine_preds.reshape(-1, 1), scaler_path).reshape(fine_preds.shape)
y_true = inverse_transform(Y_test.cpu().numpy().reshape(-1, 1), scaler_path).reshape(Y_test.shape)

# Plot comparison of first three samples
plt.figure(figsize=(15, 9))
time_steps = np.arange(num_steps)

for i in range(3):
    plt.subplot(3, 1, i + 1)

    # Plot actual values
    plt.plot(time_steps, y_true[i],
             color='#1f77b4',
             linewidth=1.5,
             label='Actual')

    # Plot base model predictions
    plt.plot(time_steps, base_preds[i],
             color='#ff7f0e',
             linestyle='--',
             linewidth=1.2,
             alpha=0.7,
             label='Base Model')

    # Plot fine-tuned model predictions
    plt.plot(time_steps, fine_preds[i],
             color='#2ca02c',
             linestyle='-.',
             linewidth=1.2,
             alpha=0.7,
             label='Fine-tuned')

    plt.title(f'Sample {i + 1} Prediction Comparison (Time Steps: {num_steps})')
    plt.xlabel('Time Step')
    plt.ylabel('Concentration')
    plt.grid(alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()

# Plot sampled point comparison (one point every 600 steps)
sample_points = np.arange(600, num_steps, 600)

plt.figure(figsize=(15, 9))
for i in range(3):
    plt.subplot(3, 1, i + 1)

    # Plot sampled actual values
    plt.plot(sample_points, y_true[i, sample_points],
             'o', color='#1f77b4',
             markersize=6,
             label='Actual Samples')

    # Plot sampled base model predictions
    plt.plot(sample_points, base_preds[i, sample_points],
             's', color='#ff7f0e',
             markersize=5,
             alpha=0.7,
             label='Base Samples')

    # Plot sampled fine-tuned model predictions
    plt.plot(sample_points, fine_preds[i, sample_points],
             '^', color='#2ca02c',
             markersize=5,
             alpha=0.7,
             label='Fine-tuned Samples')

    plt.title(f'Sample {i + 1} Stable Points Comparison ({len(sample_points)} Points)')
    plt.xlabel('Time Step')
    plt.ylabel('Concentration')
    plt.grid(alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()