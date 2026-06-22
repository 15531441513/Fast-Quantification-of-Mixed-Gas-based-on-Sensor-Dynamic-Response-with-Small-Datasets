# Fast Quantification of Mixed Gas Based on Sensor Dynamic Response with Small Datasets

This project uses deep learning models (GRU and Mamba) to analyze sensor dynamic response signals, enabling fast and accurate quantification of mixed gas concentrations with limited training data.

## Project Structure

```
├── data/              # Raw sensor response data (CSV)
│ 
├── GRU/               # GRU model implementation
│   ├── GRUmain.py     # Main training script
│   ├── GRUdata.py     # Data preprocessing
│   ├── generatebaseline.py
│   └── GRUdraw/       # Visualization and evaluation
├── MAMBA/             # Mamba state space model implementation
│   ├── mamba.py       # Main training script
│   └── requirements.txt
└── README.md
```

## Models

- **GRU**: Gated Recurrent Unit network for time-series gas quantification
- **Mamba**: State space model-based approach

## Requirements

- Python >= 3.9
- PyTorch >= 1.12.0
- numpy, pandas, h5py, joblib, scikit-learn, tqdm, matplotlib

## Usage

### GRU
```bash
cd GRU
python GRUmain.py
```

### Mamba
```bash
cd MAMBA
pip install -r requirements.txt
python mamba.py
```

## Data

69 raw sensor response data are stored in CSV files in the `data/`  directory. During the training process, the preprocessed data will be saved in HDF5 format (`.h5`). Due to file size limitations, some large files and processed files are included in releases