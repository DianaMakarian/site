from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)
CORS(app)

# === физические константы ===
G = 6.67430e-11
M_sun = 1.98847e30
R_sun = 6.957e8
AU = 1.496e11
R_earth = 6.371e6
T_sun = 5772

# === CNN Модель (для новичка) ===
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
            nn.Linear(channels[-1], 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dense(x)
        return x

# === Данные для CNN ===
df = pd.read_csv("data/KOI.csv", sep=",", skiprows=53)
mask = df["koi_pdisposition"].isin(["CANDIDATE", "FALSE POSITIVE"])
data = df[mask].copy()

phys_cols = ["koi_period", "koi_duration", "koi_depth",
             "koi_steff", "koi_srad", "koi_impact", "koi_model_snr"]
data_phys = data[phys_cols].copy().dropna(subset=phys_cols)
data = data.loc[data_phys.index].copy()

# клипы
data_phys["koi_period"] = np.clip(data_phys["koi_period"], 0.1, np.inf)
data_phys["koi_duration"] = np.clip(data_phys["koi_duration"], 0.1, np.inf)
data_phys["koi_depth"] = np.clip(data_phys["koi_depth"], 0, np.inf)
data_phys["koi_steff"] = np.clip(data_phys["koi_steff"], 2000, 10000)
data_phys["koi_srad"] = np.clip(data_phys["koi_srad"], 0.1, 10)
data_phys["koi_impact"] = np.clip(data_phys["koi_impact"], 0, 2)
data_phys["koi_model_snr"] = np.clip(data_phys["koi_model_snr"], 0, 1000)

# derived
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
L_ratio = (data_phys["koi_srad"]**2) * ((data_phys["koi_steff"]/T_sun)**4)
data_phys["insol"] = np.clip(L_ratio / (data_phys["a_AU"]**2), 0, 1e6)

features = ['koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
            'koi_model_snr', 'teq_derived', 'prad_srad_ratio',
            'koi_steff', 'koi_srad', 'insol']
data_phys = data_phys[features]

# скейлер
scaler = MinMaxScaler()
scaler.fit(data_phys.values)

# загрузка CNN модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNN_M3().to(device)
cnn_model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
cnn_model.eval()

# === Stacking модель (для учёного) ===
stack_model = joblib.load("models/stacking_model.pkl")
stack_features = ["koi_period","koi_duration","koi_depth",
                  "koi_prad","koi_teq","koi_insol",
                  "koi_steff","koi_srad","koi_kepmag"]

# ---------- API ----------

@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def serve_index():
    return send_file("index.html")

@app.route("/presets", methods=["GET"])
def get_presets():
    presets = data_phys.head(10).to_dict(orient="records")
    for i, preset in enumerate(presets):
        preset["kepler_name"] = (
            data.iloc[i]["kepler_name"] if pd.notna(data.iloc[i]["kepler_name"]) else f"Preset {i+1}"
        )
    return jsonify(presets)

@app.route("/predict", methods=["POST"])
def predict():
    data_in = request.json
    df_new = pd.DataFrame([data_in])

    # derived
    df_new["koi_smass"] = np.clip(df_new["koi_srad"] ** 0.9, 0.1, 10)
    P_sec = df_new["koi_period"] * 24 * 3600
    M_star = df_new["koi_smass"] * M_sun
    a = ((G * M_star * P_sec**2) / (4 * np.pi**2)) ** (1/3)
    df_new["a_AU"] = np.clip(a / AU, 0.001, 100)
    deltaF = np.clip(df_new["koi_depth"] * 1e-6, 0, 1)
    R_p = np.sqrt(deltaF) * df_new["koi_srad"] * (R_sun / R_earth)
    df_new["R_from_depth"] = np.clip(R_p, 0.1, 100)
    df_new["prad_srad_ratio"] = df_new["R_from_depth"] / df_new["koi_srad"]
    df_new["teq_derived"] = df_new["koi_steff"] * np.sqrt(df_new["koi_srad"] / (2 * df_new["a_AU"]))
    L_ratio = (df_new["koi_srad"]**2) * ((df_new["koi_steff"]/T_sun)**4)
    df_new["insol"] = np.clip(L_ratio / (df_new["a_AU"]**2), 0, 1e6)

    X_new = df_new[features].values
    X_new = scaler.transform(X_new)
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        prob = cnn_model(X_new_tensor).squeeze().cpu().numpy()

    return jsonify({"prob": float(prob)})

@app.route("/predict_scientist", methods=["POST"])
def predict_scientist():
    data_in = request.json
    df_new = pd.DataFrame([data_in])
    df_new = df_new[stack_features]
    pred = stack_model.predict(df_new)[0]
    label = "CONFIRMED" if pred == 1 else "CANDIDATE"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)