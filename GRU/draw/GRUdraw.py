import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
import joblib
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import pandas as pd
# --------------------- Configuration parameters ---------------------
# Parameter configuration
No = 2
case = 0
HidNeuron = 20
num_steps = 12470
batch_size = 20
Epo = 400
neuron = 14
GAS=['PdNi','Polyaniline','Nano-Te']
device = torch.device("cpu")

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

    model = FineTunedGRU(
        input_size=4,
        hidden_size=hidden_size
    ).to(device)

    model.load_state_dict(torch.load(model_path))


    for param in model.gru.parameters():
        param.requires_grad = False

    model.eval()
    return model


# --------------------- Data loading ---------------------
def load_dataset():
    """Load test dataset"""
    # Original data path
    hdf5_file_path = 'setD0'  + 'conallrandom.h5'

    # Read HDF5 data (keep original dimension order)
    with h5py.File(hdf5_file_path, 'r') as f:
        # Dimension order: (samples, time_steps, features)
        X_train = torch.tensor(f['X_train'][:, :, [0, 1, 2, 3 + No]], dtype=torch.float32)  # [batch, steps, features]
        Y_train = torch.tensor(f['Y_train'][:, :, No], dtype=torch.float32)  # [batch, steps]

        X_valid = torch.tensor(f['X_valid'][:, :, [0, 1, 2, 3 + No]], dtype=torch.float32)
        Y_valid = torch.tensor(f['Y_valid'][:, :, No], dtype=torch.float32)

        X_test = torch.tensor(f['X_test'][:, :, [0, 1, 2, 3 + No]], dtype=torch.float32)
        Y_test = torch.tensor(f['Y_test'][:, :, No], dtype=torch.float32)
        return X_test,Y_test

# --------------------- Prediction execution ---------------------
def execute_pipeline():
    # --------------------- Usage example ---------------------
    # Load fine-tuned model
    pipeline = load_finetuned_model(
        model_path=f'modelD{case}{No}1_fine.pth',
        hidden_size=HidNeuron
    ).to(device)
    print(pipeline)

    # Load data
    X_test, Y_test = load_dataset()

    # Run prediction
    with torch.no_grad():
        predictions = pipeline(X_test)

    return predictions, Y_test


# --------------------- Result visualization ---------------------
def visualize_results(predictions, targets):
    plt.figure(figsize=(15, 6))
    print(f'predictions.shape{predictions.shape}')
    print(targets.shape)
    # Plot first 3 samples
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(targets[i].cpu(), 'b-', alpha=0.6, label='True')
        plt.plot(predictions[i].cpu(), 'r--', alpha=0.8, label='Predicted')
        plt.ylabel(f'Sample {i + 1}')
        plt.grid(True)
        if i == 0:
            plt.legend()
            plt.title('Time Series Prediction Comparison')

    plt.xlabel('Time Steps')
    plt.tight_layout()
    plt.show()

    scaler_name = f'yscalarD{case}{No}1.joblib'
    scaler = joblib.load(scaler_name)
    print(scaler_name)

    targets = scaler.inverse_transform(targets[:, :])
    predictions = scaler.inverse_transform(predictions[:, :])

    plt.figure(figsize=(15, 9))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(targets[i], c='#1f77b4', alpha=0.7, linewidth=0.8, label='Actual')
        plt.plot(predictions[i], c='#ff7f0e', alpha=0.7, linewidth=0.8, label='Prediction')
        plt.title(f'Sample {i + 1} - Time Series (18000 step)')
        plt.xlabel('Timestep')
        plt.ylabel('Output')
        plt.legend()
    plt.tight_layout()
    plt.show()


    sample_indices = np.arange(0, num_steps, 1)  
    global_sampled_targets = np.zeros([1, num_steps])
    global_sampled_preds = np.zeros([1, num_steps])


    global_sampled_targets = np.concatenate(
        [global_sampled_targets, targets[:, sample_indices]],
        axis=0
    )

    global_sampled_preds = np.concatenate(
        [global_sampled_preds, predictions[:, sample_indices]],
        axis=0
    )
    print(global_sampled_targets.shape)
    global_sampled_targets = global_sampled_targets[1:, :]
    global_sampled_preds = global_sampled_preds[1:, :]

    global_sampled_targets = scaler.inverse_transform(global_sampled_targets[:, :])
    global_sampled_preds = scaler.inverse_transform(global_sampled_preds[:, :])

    sampled_targets = global_sampled_targets[:, :]  # [3, 30]
    sampled_preds = global_sampled_preds[:, :]  # [3, 30]

    plt.figure(figsize=(15, 9))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(sample_indices, sampled_targets[i], 'o-', color='#2ca02c',
                 markersize=5, label='Actual')
        plt.plot(sample_indices, sampled_preds[i], 's--', color='#d62728',
                 markersize=4, label='Prediction')
        plt.title(f'Sample {i + 1} - Stable Point (600 step per point)')
        plt.xlabel('Timestep')
        plt.ylabel('Output')
        plt.legend()
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------- Error analysis ---------------------

    # Compute relative error matrix

    relative_errors_raw = (global_sampled_preds - global_sampled_targets) / global_sampled_targets
    flat_errors_raw = relative_errors_raw.flatten()  # Flatten all error points

    # Statistical calculation
    mu_raw = np.nanmean(flat_errors_raw)  # Handle NaN values
    sigma_raw = np.nanstd(flat_errors_raw)
    bounds_raw = [mu_raw - 3 * sigma_raw, mu_raw + 3 * sigma_raw]

    # Visualization settings
    plt.figure(figsize=(14, 7))
    n, bins, _ = plt.hist(flat_errors_raw, bins='auto', density=False,
                          color='#1f77b4', alpha=0.7,
                          edgecolor='black', linewidth=0.5,
                          label='All Errors')

    # Annotate statistical lines
    plt.axvline(mu_raw, color='r', linestyle='--', linewidth=2,
                label=f'Mean: {mu_raw * 100:.2f}%')
    plt.axvline(bounds_raw[0], color='g', linestyle=':', linewidth=2,
                label=f'-3σ: {bounds_raw[0] * 100:.2f}%')
    plt.axvline(bounds_raw[1], color='g', linestyle=':', linewidth=2,
                label=f'+3σ: {bounds_raw[1] * 100:.2f}%')

    plt.title(f'Raw Relative Error Distribution ({GAS[No]})')
    plt.xlabel('Relative Error')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

    # Print statistical report
    print('=' * 60)
    print(f'Raw Error Statistics (n={len(flat_errors_raw)})')
    print('-' * 60)
    print(f'Mean (μ)      : {mu_raw:.6f}')
    print(f'Std Dev (σ)  : {sigma_raw:.6f}')
    print(f'3σ Range     : [{bounds_raw[0]:.4f}, {bounds_raw[1]:.4f}]')
    print('=' * 60)

    #-------------------- Statistics per long sequence ------------------------
    relative_errors = relative_errors_raw[:, 280::1247]
    relative_errors = np.mean(relative_errors, axis=1)
    relative_errors = relative_errors.flatten()
  
    save_errors_to_csv(
        relative_errors,
        f"relative_errors_case{case}_gas{No}.csv"
    )

    print(relative_errors.shape)


    flat_errors = relative_errors  
    # print(flat_errors)

    # 统计计算
    mu = np.mean(flat_errors)
    sigma = np.std(flat_errors)
    bounds = [mu - 3 * sigma, mu + 3 * sigma]

    ee=np.argwhere((flat_errors >= 0.075)|(flat_errors <= -0.075))
    print(ee)

    # Visualization settings
    plt.figure(figsize=(14, 7))
    n, bins, _ = plt.hist(flat_errors, bins=10, density=True,
                          color='#17becf', alpha=0.7,
                          edgecolor='black', linewidth=0.5)

    # Overlay normal distribution curve
    x = np.linspace(bins[0], bins[-1], 300)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2,
             label=f'(μ={mu:.4f}, σ={sigma:.4f})')

    # Annotate statistical lines
    vlines = [
        (mu, 'Mean', ':', '#ff0000', f'{mu * 100:.2f}%'),
        (bounds[0], '-3σ', ':', '#9467bd', f'{bounds[0] * 100:.2f}%'),
        (bounds[1], '+3σ', ':', '#9467bd', f'{bounds[1] * 100:.2f}%')
    ]

    for val, label, ls, color, percent_label in vlines:
        plt.axvline(val,
                    color=color,
                    linestyle=ls,
                    linewidth=2,
                    label=f'{label}: {percent_label}')

    plt.title('Relative Error of Stable Value of ' + str(GAS[No]))
    plt.xlabel('Relative Error')
    plt.ylabel('Number')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

    # Print statistical report
    print('=' * 60)
    print(f'| NAME        | VALUE            | AREA             |')
    print('-' * 60)
    print(f'| MEAN (μ)     | {mu:12.6f}  |                      |')
    print(f'| STD(σ)   | {sigma:12.6f}  |                      |')
    print(f'| 99.7%    | [{bounds[0] * 100:.4f}%, {bounds[1] * 100:.4f}%] | μ±3σ          |')
    print('=' * 60)

    # ===================== Filtered Analysis =====================
    # Filter MAMBA using 3sigma bounds
    filtered_errors = np.delete(flat_errors, ee)
    print(filtered_errors)
    # Recalculate statistics
    mu_filtered = np.mean(filtered_errors)
    sigma_filtered = np.std(filtered_errors)
    print(sigma_filtered)
    print(mu_filtered)
    bounds_filtered = [mu_filtered - 3 * sigma_filtered, mu_filtered + 3 * sigma_filtered]

    # Array
    data = filtered_errors  # Normal distribution data

    # Generate QQ plot
    stats.probplot(data, dist="norm", plot=plt)
    plt.title("QQ-Plot (Normal Distribution)")
    plt.show()
    # Create new figure to avoid overlap
    plt.figure(figsize=(14, 7))

    # Plot filtered histogram
    n_filtered, bins_filtered, _ = plt.hist(filtered_errors,
                                            bins='auto',
                                            density=True,
                                            color='#2ca02c',  # Different color for distinction
                                            alpha=0.7,
                                            edgecolor='black',
                                            linewidth=0.5)

    # Overlay new normal distribution curve
    x_filtered = np.linspace(bins_filtered[0], bins_filtered[-1], 300)
    plt.plot(x_filtered, norm.pdf(x_filtered, mu_filtered, sigma_filtered),
             'm-',  # Magenta curve
             lw=2,
             label=f'(Filtered μ={mu_filtered:.4f}, σ={sigma_filtered:.4f})')

    # Annotate new statistics
    vlines_filtered = [
        (mu_filtered, 'New Mean', ':', '#d62728', f'{mu_filtered * 100:.2f}%'),
        (bounds_filtered[0], 'New -3σ', '--', '#8c564b', f'{bounds_filtered[0] * 100:.2f}%'),
        (bounds_filtered[1], 'New +3σ', '--', '#8c564b', f'{bounds_filtered[1] * 100:.2f}%')
    ]

    for val, label, ls, color, percent_label in vlines_filtered:
        plt.axvline(val,
                    color=color,
                    linestyle=ls,
                    linewidth=2,
                    label=f'{label}: {percent_label}')

    plt.title('3σ Filtered Relative Error Distribution of Stable Value (' + str(GAS[No]) + ')')
    plt.xlabel('Relative Error')
    plt.ylabel('Sample Count')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

    # New statistical report
    print('\n' + '=' * 60)
    print('| Metric            | Value           | Range             |')
    print('-' * 60)
    print(f'| Mean (μ)         | {mu_filtered:12.6f}      |                  |')
    print(f'| Std Dev (σ)      | {sigma_filtered:12.6f}    |                  |')
    print(f'| 99.7% Range      | [{bounds_filtered[0] * 100:.4f}%, {bounds_filtered[1] * 100:.4f}%] | μ±3σ          |')
    print('=' * 60)


# ===================== Save relative errors to CSV file =====================
def save_errors_to_csv(errors, filename):
    """
    Save relative error matrix to CSV file

    Parameters:
    errors -- 2D array of shape (samples, time steps)
    filename -- Output file name
    """
    # Create DataFrame with index
    df = pd.DataFrame(errors)
    print(errors.shape)
    # # Add row and column names
    # df.columns = [f"Time_{i}" for i in range(errors.shape[1])]
    # df.index = [f"Sample_{i}" for i in range(errors.shape[0])]

    # # Add statistics
    # df["Mean_Error"] = np.nanmean(errors, axis=1)
    # df["Std_Dev"] = np.nanstd(errors, axis=1)
    # df["Max_Error"] = np.nanmax(errors, axis=1)
    # df["Min_Error"] = np.nanmin(errors, axis=1)

    # Save to CSV file
    df.to_csv(filename)
    print(f"Successfully saved relative error data to: {filename}")
    # print(f"File contains {errors.shape[0]} samples, each with {errors.shape[1]} time step error data")
    # print(f"Total data points: {errors.shape[0] * errors.shape[1]}")

# --------------------- Main program ---------------------
if __name__ == "__main__":
    preds, y_true = execute_pipeline()

    # Compute evaluation metrics
    mse = nn.MSELoss()(preds, y_true).item()
    # print(f'Prediction MSE: {mse:.4f}')

    # Visualize results
    visualize_results(preds, y_true)
