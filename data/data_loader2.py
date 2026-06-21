import h5py
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

data_dir = '/home/Disk3/ZXJ/final/GRU生成数据/2000样本结果/'
case = 0
D = ['D0']
h5_input = data_dir + f'{D[case]}conallrandom.h5'
h5_output = data_dir + f'set{D[case]}conallrandom.h5'


with h5py.File(h5_input, 'r') as f:
    keys = list(f.keys())
    print("原始HDF5中的数据集:", keys)
    if 'X' in keys and 'Y' in keys:
        X_raw = np.array(f['X'])
        Y_raw = np.array(f['Y'])
    elif 'x' in keys and 'y' in keys:
        X_raw = np.array(f['x'])
        Y_raw = np.array(f['y'])
    else:
        raise KeyError("未找到 'X'/'Y' 或 'x'/'y' 数据集")

sample, seq_len, x_channels = X_raw.shape
gas_num = Y_raw.shape[2]
print(f"加载数据: X {X_raw.shape}, Y {Y_raw.shape}")

x_scalers = [joblib.load(f'{data_dir}xscalar{D[case]}{i}1.joblib') for i in range(x_channels)]
y_scalers = [joblib.load(f'{data_dir}yscalar{D[case]}{i}1.joblib') for i in range(gas_num)]

X_norm = np.zeros_like(X_raw)
for i in range(x_channels):
    X_norm[:, :, i] = x_scalers[i].transform(X_raw[:, :, i].reshape(-1, 1)).reshape(sample, seq_len)

Y_norm = np.zeros_like(Y_raw)
for i in range(gas_num):
    Y_norm[:, :, i] = y_scalers[i].transform(Y_raw[:, :, i].reshape(-1, 1)).reshape(sample, seq_len)

X_train, X_temp, Y_train, Y_temp = train_test_split(X_norm, Y_norm, train_size=0.7, random_state=42)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print(f"训练集: {X_train.shape}, {Y_train.shape}")
print(f"验证集: {X_valid.shape}, {Y_valid.shape}")
print(f"测试集: {X_test.shape}, {Y_test.shape}")

with h5py.File(h5_output, 'w') as f:
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('Y_train', data=Y_train)
    f.create_dataset('X_valid', data=X_valid)
    f.create_dataset('Y_valid', data=Y_valid)
    f.create_dataset('X_test', data=X_test)
    f.create_dataset('Y_test', data=Y_test)
print(f"成功保存至: {h5_output}")