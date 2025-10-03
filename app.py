from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# Constants
G = 6.67430e-11
M_sun = 1.98847e30
R_sun = 6.957e8
AU = 1.496e11
R_earth = 6.371e6
T_sun = 5772

# CNN M3 Model
class CNN_M3(nn.Module):
    def __init__(self, input_channels=1, num_features=10):
        super(CNN_M3, self).__init__()
        channels = [16, 32, 64, 128, 256, 512]
        self.conv_blocks = nn.ModuleList()
        in_ch = input_channels
        for i, out_ch in enumerate(channels):
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
            bn = nn.BatchNorm1d(out_ch)
            act = nn.ReLU()
            drop = nn.Dropout(0.3) if (i % 2 == 1) else nn.Identity()
            self.conv_blocks.append(nn.Sequential(conv, bn, act, drop))
            in_ch = out_ch
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Sequential(
            nn.Linear(channels[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dense(x)
        return x

# Load dataset for presets
df = pd.read_csv('KOI.csv', sep=',', skiprows=53)
mask = df["koi_pdisposition"].isin(["CANDIDATE", "FALSE POSITIVE"])
data = df[mask].copy()

# Base physical features
phys_cols = ["koi_period", "koi_duration", "koi_depth", "koi_steff", "koi_srad", "koi_impact", "koi_model_snr"]
data_phys = data[phys_cols].copy()
data_phys = data_phys.dropna(subset=phys_cols)
data = data.loc[data_phys.index].copy()

# Clip extremes
data_phys["koi_period"] = np.clip(data_phys["koi_period"].fillna(1), 0.1, np.inf)
data_phys["koi_duration"] = np.clip(data_phys["koi_duration"].fillna(1), 0.1, np.inf)
data_phys["koi_depth"] = np.clip(data_phys["koi_depth"].fillna(1), 0, np.inf)
data_phys["koi_steff"] = np.clip(data_phys["koi_steff"].fillna(5000), 2000, 10000)
data_phys["koi_srad"] = np.clip(data_phys["koi_srad"].fillna(1), 0.1, 10)
data_phys["koi_impact"] = np.clip(data_phys["koi_impact"].fillna(1), 0, 2)
data_phys["koi_model_snr"] = np.clip(data_phys["koi_model_snr"].fillna(10), 0, 1000)

# Derived features
data_phys["koi_smass"] = np.clip(data_phys["koi_srad"] ** 0.9, 0.1, 10)
P_sec = data_phys["koi_period"] * 24 * 3600
M_star = data_phys["koi_smass"] * M_sun
a = ((G * M_star * P_sec**2) / (4 * np.pi**2)) ** (1/3)
data_phys["a_AU"] = np.clip(a / AU, 0.001, 100)
deltaF = np.clip(data_phys["koi_depth"] * 1e-6, 0, 1)
R_p = np.sqrt(deltaF) * data_phys["koi_srad"] * (R_sun / R_earth)
data_phys["R_from_depth"] = np.clip(R_p, 0.1, 100)
data_phys["prad_srad_ratio"] = data_phys["R_from_depth"] / data_phys["koi_srad"]
data_phys["teq_derived"] = data_phys["koi_steff"] * np.sqrt(data_phys["koi_srad"] / (2 * data_phys["a_AU"]))
L_ratio = (data_phys["koi_srad"] ** 2) * ((data_phys["koi_steff"] / T_sun) ** 4)
data_phys["insol"] = np.clip(L_ratio / (data_phys["a_AU"] ** 2), 0, 1e6)

# Features
features = ['koi_period', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_model_snr', 
            'teq_derived', 'prad_srad_ratio', 'koi_steff', 'koi_srad', 'insol']
data_phys = data_phys[features]

# Sigma clipping
for col in features:
    mean_val = data_phys[col].mean()
    std_val = data_phys[col].std()
    data_phys = data_phys[np.abs(data_phys[col] - mean_val) <= 5 * std_val]

# Scaler
scaler = MinMaxScaler()
scaler.fit(data_phys[features].values)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_M3().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

@app.route('/presets', methods=['GET'])
def get_presets():
    presets = data_phys[features].head(10).to_dict(orient='records')
    for i, preset in enumerate(presets):
        preset['kepler_name'] = data.iloc[i]['kepler_name'] if pd.notna(data.iloc[i]['kepler_name']) else f'Preset {i+1}'
    return jsonify(presets)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df_new = pd.DataFrame([data])
    
    # Process raw features
    phys_cols = ["koi_period", "koi_duration", "koi_depth", "koi_steff", "koi_srad", "koi_impact", "koi_model_snr"]
    df_phys = df_new[phys_cols].copy()
    
    # Clip
    df_phys["koi_period"] = np.clip(df_phys["koi_period"].fillna(1), 0.1, np.inf)
    df_phys["koi_duration"] = np.clip(df_phys["koi_duration"].fillna(1), 0.1, np.inf)
    df_phys["koi_depth"] = np.clip(df_phys["koi_depth"].fillna(1), 0, np.inf)
    df_phys["koi_steff"] = np.clip(df_phys["koi_steff"].fillna(5000), 2000, 10000)
    df_phys["koi_srad"] = np.clip(df_phys["koi_srad"].fillna(1), 0.1, 10)
    df_phys["koi_impact"] = np.clip(df_phys["koi_impact"].fillna(1), 0, 2)
    df_phys["koi_model_snr"] = np.clip(df_phys["koi_model_snr"].fillna(10), 0, 1000)
    
    # Derived features
    df_phys["koi_smass"] = np.clip(df_phys["koi_srad"] ** 0.9, 0.1, 10)
    P_sec = df_phys["koi_period"] * 24 * 3600
    M_star = df_phys["koi_smass"] * M_sun
    a = ((G * M_star * P_sec**2) / (4 * np.pi**2)) ** (1/3)
    df_phys["a_AU"] = np.clip(a / AU, 0.001, 100)
    deltaF = np.clip(df_phys["koi_depth"] * 1e-6, 0, 1)
    R_p = np.sqrt(deltaF) * df_phys["koi_srad"] * (R_sun / R_earth)
    df_phys["R_from_depth"] = np.clip(R_p, 0.1, 100)
    df_phys["prad_srad_ratio"] = df_phys["R_from_depth"] / df_phys["koi_srad"]
    df_phys["teq_derived"] = df_phys["koi_steff"] * np.sqrt(df_phys["koi_srad"] / (2 * df_phys["a_AU"]))
    L_ratio = (df_phys["koi_srad"] ** 2) * ((df_phys["koi_steff"] / T_sun) ** 4)
    df_phys["insol"] = np.clip(L_ratio / (df_phys["a_AU"] ** 2), 0, 1e6)
    
    # Select features
    X_new = df_phys[features].values
    X_new = scaler.transform(X_new)
    
    # Predict
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        prob = model(X_new_tensor).squeeze().cpu().numpy()
    
    return jsonify({'prob': float(prob)})

if __name__ == '__main__':
    app.run(debug=True)