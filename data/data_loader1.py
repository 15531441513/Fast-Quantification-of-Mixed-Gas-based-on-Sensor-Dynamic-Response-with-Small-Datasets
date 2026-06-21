import pandas as pd
import torch
import numpy as np
import h5py
import os
import joblib
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
print(torch.__version__)

file_path= '/final/data/'
n=10
sampcut=1
lent = 1247
randonseies=round(lent/sampcut)*n
randonresults=round(lent/sampcut)*n
enlarge=0
channelchosen=3
gas_num = 3
D=['D0','D1','D2','D3','D4','D5']
case=0
No=0

x_file = [f for f in os.listdir(file_path) if f.endswith('.csv')]
print(len(x_file))
R0=[1500,60000,1000,1750]
Datalist = np.zeros([1,round(lent/sampcut),channelchosen + gas_num])
base=np.zeros((1,gas_num))
print(Datalist.shape)
for x_f in x_file:
    print(x_f)
    filename = file_path + x_f
    FileX = pd.read_csv(filename, header=None)
    FileX = FileX.to_numpy()
    FileX = FileX[0::sampcut, 1:].astype(float)

    baseline = FileX[0, 0:gas_num]
    BL = np.tile(baseline, (FileX.shape[0], 1))
    df = np.concatenate((FileX[:, :gas_num], BL), axis=1)

    base0 = np.expand_dims(FileX[4, :gas_num], axis=0)
    base = np.concatenate((base, base0), axis=0)

    df = np.expand_dims(df, axis=0)
    Datalist = np.concatenate((Datalist, df), axis=0)

Datalist = Datalist[1:,:,:]
base=base[1:,:]
print("原始数据集维度 [样本,时序,3浓度+3基线]:", Datalist.shape)

save_scaler_dir = '/home/Disk3/ZXJ/final/GRU生成数据/2000样本结果'
for group in range(channelchosen):
    mm1 = MinMaxScaler()
    x_sub = Datalist[:, :, gas_num + group]
    a = x_sub.reshape(-1, 1)
    mm1 = mm1.fit(a)
    joblib.dump(mm1, os.path.join(save_scaler_dir, f'xscalar{D[case]}{group}1.joblib'))

for group in range(gas_num):
    mm2 = MinMaxScaler()
    y_sub = Datalist[:, :, group]
    b = y_sub.reshape(-1, 1)
    mm2 = mm2.fit(b)
    joblib.dump(mm2, os.path.join(save_scaler_dir, f'yscalar{D[case]}{group}1.joblib'))

sample=2000
for mode in range(1):
    file =np.arange(len(x_file))
    hdf5_file_path = os.path.join(save_scaler_dir, f'{D[case]}conallrandom.h5')

    tmpx=np.zeros([1,randonseies,channelchosen])
    tmpy=np.zeros([1,randonresults+enlarge,gas_num])

    file_copy = np.copy(file)
    print("总原始文件数:", len(file))

    for count in range(sample):
        print(f"生成样本 {count+1}/{sample}")
        dataset = np.zeros([1, channelchosen + gas_num])
        for t in range(n):
            random_element = np.random.randint(0, len(file))
            df = Datalist[file[random_element], :, :]
            dataset = np.append(dataset, df, axis=0)
            file = np.delete(file, random_element)
            if len(file) == 0:
                file = np.copy(file_copy)
        tmpy_sub = np.concatenate((np.tile(dataset[1, 0:gas_num], (enlarge, 1)), dataset[1:, 0:gas_num]))
        tmpx_sub = np.concatenate((np.zeros([enlarge, channelchosen]), dataset[1:, gas_num:gas_num+channelchosen]))
        tmpy_sub = np.expand_dims(tmpy_sub, axis=0)
        tmpx_sub = np.expand_dims(tmpx_sub, axis=0)

        tmpx = np.concatenate((tmpx, tmpx_sub), axis=0)
        tmpy = np.concatenate((tmpy, tmpy_sub))

tmpx = tmpx[1:, :, :]
tmpy = tmpy[1:, :, :]
print("最终X数据集 [样本,长时序,3基线]:", tmpx.shape)
print("最终Y数据集 [样本,长时序,3气体]:", tmpy.shape)

with h5py.File(hdf5_file_path, 'w') as h5f:
    h5f.create_dataset('X', data=tmpx[:,:,:])
    h5f.create_dataset('Y', data=tmpy[:,:,:])

print(f"HDF5数据集保存完成：{hdf5_file_path}")
print(f"X shape: {tmpx.shape}, Y shape: {tmpy.shape}")
