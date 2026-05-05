
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "freelancer_dataset_performant.csv"

FEATURES = [
    "avg_skills", "avg_rating", "avg_interview", "avg_training",
    "experience", "n_applications", "acceptance_rate",
    "avg_skill_gap", "avg_rate", "avg_budget"
]


def _derive_training_label(df_in: pd.DataFrame) -> pd.Series:
    """
    Construit une cible de type formation à partir du profil.
    0: Aucune, 1: Technique, 2: Soft skills, 3: Complète.
    """
    skill_gap = np.clip(1.0 - df_in["skills_match"].astype(float), 0, 1)
    tech_need = ((df_in["skills_match"] < 0.50) | (df_in["training_score"] < 60)).astype(int)
    soft_need = ((df_in["interview_score"] < 60) | (df_in["previous_rating"] < 3.5)).astype(int)
    heavy_gap = (skill_gap > 0.45).astype(int)

    y = np.zeros(len(df_in), dtype=int)
    y[(tech_need == 1) & (soft_need == 0)] = 1
    y[(tech_need == 0) & (soft_need == 1)] = 2
    y[(heavy_gap == 1) | ((tech_need == 1) & (soft_need == 1))] = 3
    return pd.Series(y)


def _prepare_training_frame(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df_raw.copy()
    df["n_applications"] = 1.0
    df["acceptance_rate"] = df["accepted"].astype(float)
    df["avg_skill_gap"] = np.clip(1.0 - df["skills_match"].astype(float), 0, 1)
    df["avg_rate"] = df["freelancer_rate"].astype(float)
    df["avg_budget"] = df["project_budget"].astype(float)
    df["avg_skills"] = df["skills_match"].astype(float)
    df["avg_rating"] = df["previous_rating"].astype(float)
    df["avg_interview"] = df["interview_score"].astype(float)
    df["avg_training"] = df["training_score"].astype(float)
    df["experience"] = df["experience_years"].astype(float)

    X = df[FEATURES].copy()
    y = _derive_training_label(df)
    return X, y


def _train_and_save_models():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable pour entraîner les modèles: {DATA_PATH}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df_train = pd.read_csv(DATA_PATH)
    X, y = _prepare_training_frame(df_train)
    X_tr, X_te, y_tr, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler_local = StandardScaler()
    X_tr_s = scaler_local.fit_transform(X_tr)

    rf_local = RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced"
    )
    rf_local.fit(X_tr_s, y_tr)

    xgb_local = None
    if XGBClassifier is not None:
        xgb_local = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=4,
            random_state=42,
            eval_metric="mlogloss",
        )
        xgb_local.fit(X_tr_s, y_tr)

    joblib.dump(scaler_local, MODELS_DIR / "scaler.pkl")
    joblib.dump(rf_local, MODELS_DIR / "rf_model.pkl")
    if xgb_local is not None:
        joblib.dump(xgb_local, MODELS_DIR / "xgb_model.pkl")

    return scaler_local, rf_local, xgb_local


def _load_or_build_models():
    scaler_path = MODELS_DIR / "scaler.pkl"
    rf_path = MODELS_DIR / "rf_model.pkl"
    xgb_path = MODELS_DIR / "xgb_model.pkl"

    if scaler_path.exists() and rf_path.exists():
        scaler_local = joblib.load(scaler_path)
        rf_local = joblib.load(rf_path)
        xgb_local = joblib.load(xgb_path) if xgb_path.exists() else None
        return scaler_local, rf_local, xgb_local

    return _train_and_save_models()


# Chargement des modèles (avec fallback auto-entraînement si les .pkl manquent)
scaler, rf_model, xgb_model = _load_or_build_models()

LABELS = {
    0: "Aucune formation nécessaire",
    1: "Formation Technique",
    2: "Formation Soft Skills",
    3: "Formation Complète"
}

FORMATIONS = {
    0: [],
    1: ["Cours développement web (React, Python)", "Formation Data Science",
        "Certification Cloud (AWS/GCP/Azure)", "Formation DevOps"],
    2: ["Communication professionnelle", "Techniques de pitch",
        "Gestion de projet", "Négociation & clients"],
    3: ["Bootcamp intensif reconversion", "Mentorat personnalisé",
        "Formation technique 6-12 semaines", "Coaching soft skills",
        "Portfolio de projets (GitHub, Kaggle)"]
}

HTML = '''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recommandation de Formation Freelancer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #2d3748; }
  .container { max-width: 780px; margin: 40px auto; padding: 0 20px; }
  h1 { text-align:center; color: #6B4E71; margin-bottom: 8px; font-size: 2rem; }
  .subtitle { text-align:center; color:#718096; margin-bottom:32px; }
  .card { background: white; border-radius: 14px; padding: 32px;
          box-shadow: 0 4px 20px rgba(0,0,0,.08); }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 24px; }
  label { display: block; font-size: .85rem; font-weight:600; color:#4a5568; margin-bottom:5px; }
  input, select { width:100%; padding:10px 12px; border:1.5px solid #e2e8f0;
                  border-radius:8px; font-size:.95rem; transition:border .2s; }
  input:focus, select:focus { outline:none; border-color:#6B4E71; }
  button { width:100%; padding:14px; background: linear-gradient(135deg,#6B4E71,#C1666B);
           color:white; border:none; border-radius:10px; font-size:1.05rem;
           font-weight:600; cursor:pointer; transition:opacity .2s; }
  button:hover { opacity:.9; }
  .result { margin-top:28px; padding:24px; border-radius:12px;
            background: #f7fafc; border-left: 5px solid #6B4E71; display:none; }
  .result h2 { color:#6B4E71; margin-bottom:10px; }
  .badge { display:inline-block; padding:4px 12px; border-radius:20px;
           font-size:.8rem; font-weight:600; margin-bottom:12px; }
  .badge-0 { background:#c6f6d5; color:#276749; }
  .badge-1 { background:#bee3f8; color:#2b6cb0; }
  .badge-2 { background:#feebc8; color:#c05621; }
  .badge-3 { background:#fed7d7; color:#c53030; }
  ul { list-style:none; padding:0; }
  ul li { padding:8px 0; border-bottom:1px solid #e2e8f0; display:flex; align-items:center; gap:8px; }
  ul li::before { content:'→'; color:#6B4E71; font-weight:bold; }
  .conf-bar { margin-top:16px; }
  .conf-item { display:flex; align-items:center; gap:10px; margin-bottom:6px; font-size:.85rem; }
  .conf-item .bar-bg { flex:1; background:#e2e8f0; border-radius:6px; height:12px; }
  .conf-item .bar-fill { height:12px; border-radius:6px; background:linear-gradient(90deg,#6B4E71,#C1666B); }
  .loading { text-align:center; color:#718096; font-style:italic; display:none; }
</style>
</head>
<body>
<div class="container">
  <h1>🎓 Recommandation de Formation</h1>
  <p class="subtitle">Analyse du profil freelancer · Détection des skills manquants · Recommandation personnalisée</p>
  <div class="card">
    <div class="grid">
      <div>
        <label>Correspondance Skills (0-1)</label>
        <input type="number" id="avg_skills" min="0" max="1" step="0.01" value="0.4" placeholder="0.0 - 1.0">
      </div>
      <div>
        <label>Note précédente (1-5)</label>
        <input type="number" id="avg_rating" min="1" max="5" step="0.1" value="3.5">
      </div>
      <div>
        <label>Score entretien (0-100)</label>
        <input type="number" id="avg_interview" min="0" max="100" step="0.1" value="55">
      </div>
      <div>
        <label>Score formation (0-100)</label>
        <input type="number" id="avg_training" min="0" max="100" step="0.1" value="50">
      </div>
      <div>
        <label>Années d'expérience</label>
        <input type="number" id="experience" min="0" max="40" step="1" value="3">
      </div>
      <div>
        <label>Nombre de candidatures</label>
        <input type="number" id="n_applications" min="1" step="1" value="10">
      </div>
      <div>
        <label>Taux d'acceptation (0-1)</label>
        <input type="number" id="acceptance_rate" min="0" max="1" step="0.01" value="0.3">
      </div>
      <div>
        <label>Gap de skills (0-1)</label>
        <input type="number" id="avg_skill_gap" min="0" max="1" step="0.01" value="0.2">
      </div>
      <div>
        <label>Taux horaire ($/h)</label>
        <input type="number" id="avg_rate" min="0" step="1" value="50">
      </div>
      <div>
        <label>Budget projet ($)</label>
        <input type="number" id="avg_budget" min="0" step="100" value="5000">
      </div>
    </div>
    <div style="margin-bottom:16px">
      <label>Modèle de prédiction</label>
      <select id="model_choice">
        <option value="rf">Random Forest (recommandé)</option>
        <option value="xgb">XGBoost</option>
      </select>
    </div>
    <button onclick="predict()">🔍 Analyser le profil & Recommander</button>
    <p class="loading" id="loading">Analyse en cours...</p>
    <div class="result" id="result">
      <span class="badge" id="badge"></span>
      <h2 id="result-title"></h2>
      <p id="result-desc" style="color:#718096; margin-bottom:14px"></p>
      <ul id="result-list"></ul>
      <div class="conf-bar" id="conf-bar"></div>
    </div>
  </div>
</div>
<script>
const LABELS = {"0":"Aucune formation","1":"Formation Technique",
                "2":"Formation Soft Skills","3":"Formation Complète"};

async function predict() {
  const data = {
    avg_skills: +document.getElementById('avg_skills').value,
    avg_rating: +document.getElementById('avg_rating').value,
    avg_interview: +document.getElementById('avg_interview').value,
    avg_training: +document.getElementById('avg_training').value,
    experience: +document.getElementById('experience').value,
    n_applications: +document.getElementById('n_applications').value,
    acceptance_rate: +document.getElementById('acceptance_rate').value,
    avg_skill_gap: +document.getElementById('avg_skill_gap').value,
    avg_rate: +document.getElementById('avg_rate').value,
    avg_budget: +document.getElementById('avg_budget').value,
    model: document.getElementById('model_choice').value
  };
  document.getElementById('loading').style.display = 'block';
  document.getElementById('result').style.display = 'none';
  const r = await fetch('/predict', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  const res = await r.json();
  document.getElementById('loading').style.display = 'none';
  document.getElementById('result').style.display = 'block';
  const badge = document.getElementById('badge');
  badge.textContent = LABELS[res.category];
  badge.className = 'badge badge-' + res.category;
  document.getElementById('result-title').textContent = res.title;
  document.getElementById('result-desc').textContent = res.description;
  const ul = document.getElementById('result-list');
  ul.innerHTML = res.formations.map(f => '<li>' + f + '</li>').join('');
  const cb = document.getElementById('conf-bar');
  cb.innerHTML = '<p style="font-weight:600;margin-bottom:8px">Confiance par catégorie :</p>';
  res.probabilities.forEach((p, i) => {
    cb.innerHTML += '<div class="conf-item"><span style="width:180px">' + LABELS[i] +
      '</span><div class="bar-bg"><div class="bar-fill" style="width:' + (p*100).toFixed(1) +
      '%"></div></div><span>' + (p*100).toFixed(1) + '%</span></div>';
  });
}
</script>
</body></html>'''  # noqa


FORMATIONS_DESC = {
    0: "Votre profil est bien aligné avec les projets disponibles.",
    1: "Vos skills techniques ne correspondent pas aux exigences des projets.",
    2: "Vos compétences relationnelles peuvent être améliorées.",
    3: "Votre profil bénéficierait d'une mise à niveau complète."
}

FORMATIONS_TITRES = {
    0: "✅ Aucune formation nécessaire",
    1: "🔧 Formation Technique Recommandée",
    2: "🤝 Formation Soft Skills Recommandée",
    3: "📚 Formation Complète Recommandée"
}

@app.route("/")
def index():
    return HTML

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    model_choice = data.get("model", "rf")
    features_input = [data.get(f, 0) for f in FEATURES]
    X_input = scaler.transform([features_input])
    
    if model_choice == "xgb" and xgb_model is not None:
        cat = int(xgb_model.predict(X_input)[0])
        probas = xgb_model.predict_proba(X_input)[0].tolist()
    else:
        cat = int(rf_model.predict(X_input)[0])
        probas = rf_model.predict_proba(X_input)[0].tolist()

    return jsonify({
        "category": cat,
        "title": FORMATIONS_TITRES[cat],
        "description": FORMATIONS_DESC[cat],
        "formations": FORMATIONS[cat],
        "probabilities": probas
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
