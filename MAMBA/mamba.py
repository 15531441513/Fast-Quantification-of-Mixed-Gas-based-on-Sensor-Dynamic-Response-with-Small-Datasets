import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from mamba_block import Mamba  # Pure PyTorch implementation (CPU-compatible)
from tqdm import tqdm
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def upsample_fft(data, original_len=40, target_len=200, axis=1):

    if data.shape[axis] != original_len:
        raise ValueError(f"Input time length {data.shape[axis]} != {original_len}")
    samples, _, features = data.shape
    upsampled = np.zeros((samples, target_len, features), dtype=data.dtype)

    half = original_len // 2  # original_len=40 
    for i in range(samples):
        for f in range(features):
            signal = data[i, :, f]
            spec = np.fft.fft(signal, n=original_len)
            new_spec = np.zeros(target_len, dtype=complex)

            new_spec[:half] = spec[:half]

            new_spec[target_len - half:] = spec[half:]
            signal_up = np.fft.ifft(new_spec)
            upsampled[i, :, f] = np.real(signal_up)
    return upsampled

# ==================== 2. Model Definition ====================
class MambaOnlyRegressor(nn.Module):
    def __init__(self, input_dim=3, d_model=256, hidden_dim=128, d_state=16, expand=2, d_conv=4):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, d_model)
        )
        self.mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Sequential(
            nn.Tanh(),
            nn.Linear(d_model, 3)
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.mamba(x)
        x = self.dropout(x)
        last_step = x[:, -1, :]
        return self.out_proj(last_step)

def calculate_mre(preds, targets):
    relative_errors = np.abs((preds - targets) / (targets + 1e-6))
    return np.mean(relative_errors, axis=0)

# ==================== 3. Main Training Function ====================
def train_with_upsample():
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    REAL_WEIGHT = 10.0
    BATCH_SIZE = 128
    EPOCHS = 1000
    LEARNING_RATE = 1e-3
    GAS_LOSS_WEIGHTS = torch.tensor([1.0, 1.5, 1.0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    sim_h5 = os.path.join(DATA_DIR, 'newdata_base_len40.h5')
    real_h5 = os.path.join(DATA_DIR, 'real_data_mamba_len40.h5')
    with h5py.File(sim_h5, 'r') as f:
        sim_X = f['X'][:]   
        sim_Y = f['Y'][:]
    with h5py.File(real_h5, 'r') as f:
        real_X = f['X'][:]   
        real_Y = f['Y'][:]

    print(f"Before upsampling: sim_X {sim_X.shape}, real_X {real_X.shape}")

    sim_X = upsample_fft(sim_X, original_len=40, target_len=200)
    real_X = upsample_fft(real_X, original_len=40, target_len=200)


    # ----- 10-fold cross validation -----
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_fold_mres = []
    all_real_preds = []
    all_real_targets = []

    gas_weights = GAS_LOSS_WEIGHTS.to(device)
    fold_pbar = tqdm(enumerate(kf.split(real_X)), total=10, desc="K-Fold Progress")

    for fold, (train_idx, val_idx) in fold_pbar:
        # Build training set
        x_train_real, y_train_real = real_X[train_idx], real_Y[train_idx]
        w_sim = np.ones(len(sim_X), dtype=np.float32)
        w_real = np.ones(len(x_train_real), dtype=np.float32) * REAL_WEIGHT
        x_train = np.concatenate([sim_X, x_train_real], axis=0)
        y_train = np.concatenate([sim_Y, y_train_real], axis=0)
        weights_train = np.concatenate([w_sim, w_real], axis=0)

        train_ds = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train),
                                 torch.FloatTensor(weights_train))
        val_ds = TensorDataset(torch.FloatTensor(real_X[val_idx]), torch.FloatTensor(real_Y[val_idx]))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = MambaOnlyRegressor().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
        criterion = nn.HuberLoss(reduction='none')

        for epoch in range(EPOCHS):
            model.train()
            for bx, by, bw in train_loader:
                bx, by, bw = bx.to(device), by.to(device), bw.to(device)
                optimizer.zero_grad()
                out = model(bx)
                weighted_loss = ((criterion(out, by) * gas_weights).mean(dim=1) * bw).mean()
                weighted_loss.backward()
                optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        fold_preds_norm, fold_targets_norm = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                preds = model(vx).cpu().numpy()
                fold_preds_norm.append(preds)
                fold_targets_norm.append(vy.numpy())

        preds_concat = np.concatenate(fold_preds_norm, axis=0)
        targets_concat = np.concatenate(fold_targets_norm, axis=0)

        # Denormalize
        denorm_preds = np.zeros_like(preds_concat)
        denorm_targets = np.zeros_like(targets_concat)
        for i in range(3):
            scaler = joblib.load(os.path.join(DATA_DIR, f'y_scaler_{i}.joblib'))
            denorm_preds[:, i] = scaler.inverse_transform(preds_concat[:, i].reshape(-1, 1)).flatten()
            denorm_targets[:, i] = scaler.inverse_transform(targets_concat[:, i].reshape(-1, 1)).flatten()

        all_real_preds.append(denorm_preds)
        all_real_targets.append(denorm_targets)

        mre = calculate_mre(denorm_preds, denorm_targets)
        all_fold_mres.append(mre)
        tqdm.write(f">> Fold {fold+1} MRE: NH3:{mre[0]:.2%}, NO2:{mre[1]:.2%}, H2:{mre[2]:.2%}")

    # Final statistics and save
    final_avg = np.mean(all_fold_mres, axis=0)
    print(f"\nFinal average MRE: NH3:{final_avg[0]:.2%}, NO2:{final_avg[1]:.2%}, H2:{final_avg[2]:.2%}")

    results_X = np.concatenate(all_real_preds, axis=0)
    results_Y = np.concatenate(all_real_targets, axis=0)
    df_results = pd.DataFrame({
        'NH3_True': results_Y[:, 0], 'NH3_Pred': results_X[:, 0],
        'NO2_True': results_Y[:, 1], 'NO2_Pred': results_X[:, 1],
        'H2_True': results_Y[:, 2], 'H2_Pred': results_X[:, 2]
    })
    save_path = os.path.join(DATA_DIR, 'results.csv')
    df_results.to_csv(save_path, index=False)
    print(f"Prediction comparison table saved to: {save_path}")

if __name__ == "__main__":
    train_with_upsample()