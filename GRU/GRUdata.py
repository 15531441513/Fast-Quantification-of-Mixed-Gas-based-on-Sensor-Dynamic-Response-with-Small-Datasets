import pandas as pd
import torch
import numpy as np
from itertools import product
import os
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shutil
import gc
from generatebaseline import DataGenerator 

# ==================== Get script directory ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------- Configuration parameters ----------------------
case = 0
HidNeuron = 20
enlarge = 0
sampcut = 1
n = 10
batch_size = 20
neuron = 14

GAS = ['H2', 'NH3', 'NO2']
device = torch.device("cpu")
R0 = [1500, 1000, 1750]

# ---------- Output directories (relative to script directory) ----------
filepath = os.path.join(BASE_DIR, 'dataset/')          # Store generated augmented samples
final_path = os.path.join(BASE_DIR, 'final_results/')  # Store inference results

# Clean and recreate dataset directory (old samples will be overwritten)
shutil.rmtree(filepath, ignore_errors=True)
os.makedirs(filepath, exist_ok=True)
os.makedirs(final_path, exist_ok=True)  # Ensure final_results exists

# ---------------------- Concentration generation (17 equal intervals) ----------------------
num_columns = 3
virture_columns = 6
lent = 1800
samplestep = lent
randonseies = round(lent / sampcut) * n
num_steps = samplestep * n
gas_empty_idx = 3
sensor_empty_idx = 1

# 17 gradients including endpoints (as described in the literature)
con_H2 = np.linspace(8000, 32000, 17)   # H2,  step 1500 ppm
con_NH3 = np.linspace(10, 50, 17)       # NH3, step 2.5 ppm
con_NO2 = np.linspace(2, 10, 17)        # NO2, step 0.5 ppm

setnum = len(con_NH3) * len(con_NO2) * len(con_H2)
print(f"Total combinations: {setnum}")  # 4913

# ---------------------- Generate all concentration combinations and signals ----------------------
arrayset = np.zeros([1, samplestep, 3])  # Placeholder

for nh3, no2, h2 in product(con_NH3, con_NO2, con_H2):
    row = np.array([h2, nh3, no2], dtype=float)
    signal = np.tile(row, (samplestep, 1))
    signal[0:3, :] = 0
    signal[int(round(1200 // sampcut + 1)):, :] = 0
    arrayset = np.concatenate((arrayset, np.expand_dims(signal, axis=0)), axis=0)

arrayset = arrayset[1:, :, :]  # (4913, 1800, 3)
print(f"Actual generated samples: {arrayset.shape[0]}")

# ---------------------- Generate baseline and save ----------------------
def generate_virtual_baseline(
        input_data: np.ndarray,
        n_steps: int = samplestep,
        random_seed: int = None,
) -> np.ndarray:
    actual_steps, num_features = input_data.shape
    if random_seed is not None:
        np.random.seed(random_seed)
    baseline_vectors = DataGenerator.generate(1)   # Ensure this function exists
    baseline_data = np.tile(baseline_vectors, (n_steps, 1))
    merged = np.concatenate([input_data, baseline_data], axis=1)
    return merged

print("Generating CSV files...")
for k in range(arrayset.shape[0]):
    df_res = arrayset[k, :, :]
    df_res = generate_virtual_baseline(df_res)
    df_res = pd.DataFrame(df_res)
    df_res.to_csv(os.path.join(filepath, f"{0 + k}.csv"), index=False, header=False)
print(f"All samples saved to: {filepath}")

# --------------------- Model definition and loading (paths relative to BASE_DIR) ---------------------
class FineTunedGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FineTunedGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(neuron, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, _ = self.gru(x)
        output = self.dropout(output)
        output = output[:, :, -neuron:]
        out = self.fc(output)
        return self.sigmoid(out.squeeze(-1))

def load_finetuned_model(model_path, hidden_size):
    model = FineTunedGRU(input_size=4, hidden_size=hidden_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    for param in model.gru.parameters():
        param.requires_grad = False
    model.eval()
    return model

def normalize_step_signals(x_long: np.ndarray, sensorchosen, gaschosen, model_prefix: str = "xscalarD0") -> np.ndarray:
    total_steps, num_features = x_long.shape
    x_trains = np.empty_like(x_long)
    for feature_idx in range(6):
        if feature_idx < 3:
            filename = f"{model_prefix}{gaschosen[feature_idx]}1.joblib"
        else:
            filename = f"{model_prefix}{4 + sensorchosen[feature_idx - 3]}1.joblib"
        scaler_path = os.path.join(BASE_DIR, filename)   # All scalers are in the script directory
        scaler = joblib.load(scaler_path)
        feature_data = x_long[:, feature_idx].reshape(-1, 1)
        x_trains[:, feature_idx] = scaler.transform(feature_data).flatten()
    return x_trains

def process_predictions(preds, original_files):
    processed_results = np.zeros([1, samplestep, virture_columns + num_columns])
    sample_pred = preds
    final_matrix = np.zeros((samplestep, 3))
    for gas_idx in range(3):
        full_sequence = sample_pred[:, gas_idx]
        for t in range(round(samplestep / sampcut)):
            start_idx = samplestep + t
            end_idx = samplestep * n
            extracted_points = full_sequence[start_idx:end_idx:samplestep]
            avg_value = np.mean(extracted_points)
            final_matrix[t, gas_idx] = avg_value
        original_data = pd.read_csv(original_files, header=None).values
        combined = np.concatenate([final_matrix, original_data[:, :]], axis=1)
        processed_results = combined
    return np.array(processed_results)

def execute_pipeline(no, xfile):
    # Model weight files are also in the script directory
    model_path = os.path.join(BASE_DIR, f"modelD{case}{no}1_fine.pth")
    pipeline = load_finetuned_model(model_path, hidden_size=HidNeuron).to(device)
    xfile = torch.FloatTensor(xfile).to(device)
    if xfile.ndim == 2:
        xfile = xfile.unsqueeze(0)
    with torch.no_grad():
        predictions = pipeline(xfile)
    return predictions.cpu().numpy()

# --------------------- Main inference pipeline ---------------------
if __name__ == "__main__":
    print("\nRunning inference pipeline...")
    file_load = filepath
    sensorchosen = [0, 2, 3]
    gaschosen = [0, 1, 2]

    x_files = sorted([f for f in os.listdir(file_load) if f.endswith('.csv')])
    print(f"Found {len(x_files)} data files")

    batch_size = 20
    total_batches = (len(x_files) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(x_files))
        current_files = x_files[start_idx:end_idx]
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches}, containing {len(current_files)} files")

        batch_data = np.zeros((1, samplestep * n, virture_columns))
        for file_idx, filename in enumerate(current_files):
            file_path = os.path.join(file_load, filename)
            fi = pd.read_csv(file_path, header=None).values.astype(float)
            temp_norm = normalize_step_signals(fi[:, 0:6], gaschosen=gaschosen, sensorchosen=sensorchosen)
            temp_long = np.vstack([temp_norm] * n)
            newfile = np.expand_dims(temp_long, axis=0)
            batch_data = np.concatenate([batch_data, newfile], axis=0)

        batch_data = batch_data[1:, :, :]
        print(f"Current batch data shape: {batch_data.shape}")

        final_preds = np.zeros((batch_data.shape[0], batch_data.shape[1], 1))
        for gas_idx in range(3):
            temp_channel = batch_data[:, :, [0, 1, 2, 3 + gas_idx]]
            array_pred = execute_pipeline(sensorchosen[gas_idx], temp_channel)
            # Inverse normalization scalers are also in the script directory
            yscaler_path = os.path.join(BASE_DIR, f"yscalarD0{sensorchosen[gas_idx]}1.joblib")
            yscaler = joblib.load(yscaler_path)
            y_preds = yscaler.inverse_transform(array_pred.reshape(-1, 1)).reshape(
                array_pred.shape[0], array_pred.shape[1], 1
            )
            final_preds = np.concatenate([final_preds, y_preds], axis=2)

        final_preds = final_preds[:, :, 1:]

        for file_order, filename in enumerate(current_files):
            original_idx = start_idx + file_order
            processed_data = process_predictions(final_preds[file_order, :, :],
                                                 os.path.join(file_load, filename))
            expanded = np.zeros((processed_data.shape[0], 8))
            expanded[:, :sensor_empty_idx] = processed_data[:, :sensor_empty_idx]
            expanded[:, sensor_empty_idx + 1:4 + gas_empty_idx] = processed_data[:, sensor_empty_idx:3 + gas_empty_idx]
            expanded[:, 4 + gas_empty_idx + 1:8] = processed_data[:, 3 + gas_empty_idx:6]
            # Save results to final_results subdirectory
            out_file = os.path.join(final_path, f"combined_{0 + original_idx}.csv")
            pd.DataFrame(expanded).to_csv(out_file, index=False, header=False)

        del batch_data, final_preds
        gc.collect()

    print(f"\nPipeline complete! Results saved to {final_path}")