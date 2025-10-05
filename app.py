from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import io

app = Flask(__name__)
CORS(app)

# === фізичні константи ===
G = 6.67430e-11
M_sun = 1.98847e30
R_sun = 6.957e8
AU = 1.496e11
R_earth = 6.371e6
T_sun = 5772

# === CNN Модель (для новачків) ===
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

# === Дані для CNN ===
df = pd.read_csv("data/KOI.csv", sep=",", skiprows=53)
mask = df["koi_pdisposition"].isin(["CANDIDATE", "FALSE POSITIVE"])
data = df[mask].copy()

phys_cols = ["koi_period", "koi_duration", "koi_depth",
             "koi_steff", "koi_srad", "koi_impact", "koi_model_snr"]
data_phys = data[phys_cols].copy().dropna(subset=phys_cols)
data = data.loc[data_phys.index].copy()

# обрізання значень
data_phys["koi_period"] = np.clip(data_phys["koi_period"], 0.1, np.inf)
data_phys["koi_duration"] = np.clip(data_phys["koi_duration"], 0.1, np.inf)
data_phys["koi_depth"] = np.clip(data_phys["koi_depth"], 0, np.inf)
data_phys["koi_steff"] = np.clip(data_phys["koi_steff"], 2000, 10000)
data_phys["koi_srad"] = np.clip(data_phys["koi_srad"], 0.1, 10)
data_phys["koi_impact"] = np.clip(data_phys["koi_impact"], 0, 2)
data_phys["koi_model_snr"] = np.clip(data_phys["koi_model_snr"], 0, 1000)

# похідні ознаки
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

scaler = MinMaxScaler()
scaler.fit(data_phys.values)

# завантаження CNN моделі
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNN_M3().to(device)
cnn_model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
cnn_model.eval()

# === Завантаження нової двоступінчатої моделі (для науковців) ===
scientist_model = joblib.load("models/stacking_model.pkl")

# Перевіряємо тип моделі
if isinstance(scientist_model, dict):
    # Нова версія - словник з компонентами
    print("[INFO] Завантажено модель у новому форматі (словник)")
    cb1 = scientist_model['stage1_model']
    iso1 = scientist_model['stage1_calibrator']
    ensembles = scientist_model['stage2_ensembles']
    calibrater = scientist_model['stage2_calibrator']
    top_features = scientist_model['top_features']
    stage2_scaler = scientist_model.get('stage2_scaler', None)
    EPSILON = scientist_model.get('epsilon', 0.07)
    stage1_required_features = scientist_model.get('stage1_features', None)
    
    print(f"[INFO] Завантажено модель з {len(ensembles)} ансамблевими моделями Stage-2")
    print(f"[INFO] Top {len(top_features)} ознак для Stage-2: {top_features}")
    USE_TWO_STAGE = True
else:
    # Стара версія - один CatBoost класифікатор
    print("[WARNING] Завантажено модель у старому форматі (один класифікатор)")
    print("[WARNING] Використовується спрощений одноступінчатий прогноз")
    simple_model = scientist_model
    # Ознаки для старої моделі
    stack_features = ["koi_period","koi_duration","koi_depth",
                      "koi_prad","koi_teq","koi_insol",
                      "koi_steff","koi_srad","koi_kepmag"]
    USE_TWO_STAGE = False

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
            data.iloc[i]["kepler_name"] if pd.notna(data.iloc[i]["kepler_name"]) else f"False Positive Preset {i+1}"
        )
    return jsonify(presets)

@app.route("/predict", methods=["POST"])
def predict():
    """Прогноз для новачків (CNN модель)"""
    data_in = request.json
    df_new = pd.DataFrame([data_in])

    # обчислення похідних ознак
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
    """
    Прогноз для науковців (двоступінчата модель model.pkl)
    Повертає 3 класи: FALSE POSITIVE, CANDIDATE, CONFIRMED
    """
    data_in = request.json
    df_new = pd.DataFrame([data_in])
    
    # Переконуємося що всі необхідні ознаки присутні
    # Якщо stage1_required_features збережені, використовуємо їх
    if stage1_required_features is not None:
        # Додаємо відсутні колонки з NaN якщо потрібно
        for col in stage1_required_features:
            if col not in df_new.columns:
                df_new[col] = np.nan
        X_stage1 = df_new[stage1_required_features].copy()
    else:
        # Інакше використовуємо всі числові колонки що є
        X_stage1 = df_new.select_dtypes(include=[np.number]).copy()
    
    # Stage-1: прогноз P(планета)
    p1_raw = cb1.predict_proba(X_stage1)[:, 1]
    P_planet = iso1.predict(p1_raw)[0]  # калібрований
    P_planet = np.clip(P_planet, 0.0, 1.0)
    
    # Stage-2: прогноз P(confirmed | планета)
    # Використовуємо тільки top_features
    X_stage2 = df_new[top_features].copy()
    
    # Якщо є scaler для Stage-2, застосовуємо його (не обов'язково)
    if stage2_scaler is not None:
        X_stage2_scaled = pd.DataFrame(
            stage2_scaler.transform(X_stage2), 
            columns=X_stage2.columns, 
            index=X_stage2.index
        )
    else:
        X_stage2_scaled = X_stage2
    
    # Ансамбль Stage-2 моделей
    ensemble_preds = []
    for model in ensembles:
        prob = model.predict_proba(X_stage2_scaled)[:, 1]
        ensemble_preds.append(prob)
    
    p2_raw_ens = np.mean(ensemble_preds, axis=0)[0]
    
    # Калібрація Platt scaling
    p2_calib = calibrater.predict_proba(np.array([[p2_raw_ens]]))[:, 1][0]
    p2_calib = np.clip(p2_calib, 0.0, 1.0)
    
    # Зона невизначеності (опціонально)
    if (0.5 - EPSILON) <= p2_calib <= (0.5 + EPSILON):
        p2_adj = 0.0  # консервативно -> CANDIDATE
    else:
        p2_adj = p2_calib
    
    # Комбінуємо обидва етапи
    P_conf = P_planet * p2_adj
    P_cand = P_planet * (1.0 - p2_adj)
    P_fp = 1.0 - P_planet
    
    # Визначаємо фінальний клас
    probs = [P_fp, P_cand, P_conf]
    pred_idx = np.argmax(probs)
    labels = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
    prediction = labels[pred_idx]
    
    return jsonify({
        "prediction": prediction,
        "probabilities": {
            "FALSE_POSITIVE": float(P_fp),
            "CANDIDATE": float(P_cand),
            "CONFIRMED": float(P_conf)
        },
        "stage1_planet_prob": float(P_planet),
        "stage2_confirmed_prob": float(p2_calib)
    })

@app.route("/save_planet", methods=["POST"])
def save_planet():
    """Зберігає планету у CSV форматі для подальшого аналізу"""
    data_in = request.json
    planet_name = data_in.get("planet_name", "Невідома")
    
    # Створюємо рядок з необхідними колонками
    new_row = {
        "kepler_name": planet_name,
        "koi_period": data_in.get("koi_period"),
        "koi_duration": data_in.get("koi_duration"),
        "koi_depth": data_in.get("koi_depth"),
        "koi_prad": data_in.get("prad_srad_ratio", 0) * data_in.get("koi_srad", 1) * (R_sun / R_earth),
        "koi_teq": data_in.get("teq_derived"),
        "koi_insol": data_in.get("insol"),
        "koi_steff": data_in.get("koi_steff"),
        "koi_srad": data_in.get("koi_srad"),
        "koi_kepmag": data_in.get("koi_kepmag", 15.0),
        "koi_impact": data_in.get("koi_impact", 0.5),
        "koi_model_snr": data_in.get("koi_model_snr", 10.0)
    }
    
    # Генеруємо CSV
    df_new = pd.DataFrame([new_row])
    csv_buffer = io.StringIO()
    df_new.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # Безпечна назва файлу
    safe_filename = "".join(c for c in planet_name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_filename:
        safe_filename = "планета"
    
    return jsonify({
        "message": f"Планета '{planet_name}' готова до завантаження!",
        "status": "success",
        "csv_data": csv_content,
        "filename": f"{safe_filename}.csv"
    })
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
