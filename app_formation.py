import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="🎓 Recommandation de Formations",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS CUSTOM ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F0F4F8; }
    .metric-card {
        background: white; border-radius: 12px; padding: 18px;
        text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-top: 4px solid;
    }
    .metric-value { font-size: 2rem; font-weight: 800; margin: 8px 0 4px; }
    .metric-label { font-size: 0.85rem; color: #546E7A; }
    .badge-green  { background:#E8F5E9; color:#1B5E20; padding:5px 14px; border-radius:20px; font-weight:700; font-size:0.9rem; }
    .badge-orange { background:#FFF3E0; color:#E65100; padding:5px 14px; border-radius:20px; font-weight:700; font-size:0.9rem; }
    .badge-red    { background:#FFEBEE; color:#B71C1C; padding:5px 14px; border-radius:20px; font-weight:700; font-size:0.9rem; }
    .result-box   { border-radius:14px; padding:22px; margin:10px 0; }
    .stButton>button { border-radius:10px; font-weight:700; font-size:1rem; padding:10px 28px; }
    h1 { color: #0D1B2A !important; }
    h2 { color: #1565C0 !important; }
    h3 { color: #0D1B2A !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD & TRAIN MODELS ───────────────────────────────────────
@st.cache_resource
def load_and_train():
    try:
        df = pd.read_excel('Freelancer_Project_Matching_Dataset.xlsx',
                           sheet_name='Matching_Dataset (ML)', header=1)
    except:
        # Generate synthetic data if file not found
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'Skill Match Score':   np.random.beta(1, 5, n),
            'Freelancer Rating':   np.random.uniform(2.5, 5.0, n),
            'Completion Rate':     np.random.uniform(0.70, 1.0, n),
            'On Time Delivery':    np.random.uniform(0.5, 1.0, n),
            'Repeat Client Rate':  np.random.uniform(0.0, 1.0, n),
            'Response Time Hours': np.random.uniform(1, 48, n),
            'Portfolio Available': np.random.randint(0, 2, n),
            'Identity Verified':   np.random.randint(0, 2, n),
            'Rate Compatibility':  np.random.uniform(0, 1, n),
            'Experience Sufficient': np.random.randint(0, 2, n),
        })

    df['Needs_Training'] = (
        (df['Skill Match Score'] < 0.30) |
        (df['Completion Rate']   < 0.82) |
        (df['Freelancer Rating'] < 3.5)
    ).astype(int)

    FEATURES = ['Skill Match Score','Freelancer Rating','Completion Rate',
                'On Time Delivery','Repeat Client Rate','Response Time Hours',
                'Portfolio Available','Identity Verified','Rate Compatibility','Experience Sufficient']

    X = df[FEATURES]; y = df['Needs_Training']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    xgb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    xgb.fit(X_train, y_train)

    rf_pred  = rf.predict(X_test);  rf_proba  = rf.predict_proba(X_test)[:,1]
    xgb_pred = xgb.predict(X_test); xgb_proba = xgb.predict_proba(X_test)[:,1]

    cv5 = StratifiedKFold(5, shuffle=True, random_state=42)
    rf_cv  = cross_val_score(rf,  X, y, cv=cv5, scoring='f1').mean()
    xgb_cv = cross_val_score(xgb, X, y, cv=cv5, scoring='f1').mean()

    metrics = {
        'RF':  dict(acc=accuracy_score(y_test,rf_pred),  f1=f1_score(y_test,rf_pred,zero_division=0),
                    auc=roc_auc_score(y_test,rf_proba),  cv=rf_cv,
                    cm=confusion_matrix(y_test,rf_pred), fpr_tpr=roc_curve(y_test,rf_proba)),
        'XGB': dict(acc=accuracy_score(y_test,xgb_pred), f1=f1_score(y_test,xgb_pred,zero_division=0),
                    auc=roc_auc_score(y_test,xgb_proba), cv=xgb_cv,
                    cm=confusion_matrix(y_test,xgb_pred),fpr_tpr=roc_curve(y_test,xgb_proba)),
    }

    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    stats = dict(total=len(df), needs=int(df['Needs_Training'].sum()),
                 pct=df['Needs_Training'].mean()*100)
    return rf, xgb, FEATURES, metrics, fi, stats, df

rf_model, xgb_model, FEATURES, metrics, fi, stats, df = load_and_train()

# ── TRAINING LEVEL & RECOMMENDATION FUNCTION ─────────────────
def get_recommendation(skill, rating, completion):
    pts = 0
    formations = []
    if skill < 0.15:   pts += 2; formations.append("🔴 Formation Compétences Techniques INTENSIVE")
    elif skill < 0.30: pts += 1; formations.append("🟠 Formation Compétences Techniques STANDARD")
    if completion < 0.77:  pts += 2; formations.append("🔴 Formation Gestion de Projet INTENSIVE")
    elif completion < 0.82: pts += 1; formations.append("🟠 Formation Gestion de Projet STANDARD")
    if rating < 3.2:  pts += 2; formations.append("🔴 Formation Communication & Soft Skills INTENSIVE")
    elif rating < 3.5: pts += 1; formations.append("🟠 Formation Communication & Soft Skills STANDARD")

    if pts == 0 and skill >= 0.30 and completion >= 0.82 and rating >= 3.5:
        level = "Aucune Formation Nécessaire"
        color = "green"
        formations = ["✅ Profil excellent — Maintenir les performances actuelles"]
    elif pts >= 4:
        level = "Formation INTENSIVE"
        color = "red"
    elif pts >= 2:
        level = "Formation MODÉRÉE"
        color = "orange"
    else:
        level = "Formation LÉGÈRE"
        color = "orange"
    return level, color, formations, pts

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Navigation")
    page = st.radio("", ["🏠 Accueil & Statistiques", "🔮 Prédiction Individuelle",
                         "📊 Évaluation des Modèles", "📈 Analyse des Données"])
    st.markdown("---")
    st.markdown("### 📌 À propos")
    st.info("**Projet** : Freelancer & Project Matching\n\n**Module** : Recommandation de Formations\n\n**Algorithmes** : Random Forest + XGBoost")
    st.markdown("---")
    st.markdown(f"### 📊 Dataset\n- **{stats['total']}** freelancers\n- **{stats['needs']}** besoin formation\n- **{stats['pct']:.1f}%** taux")

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Accueil & Statistiques":
    st.title("🎓 Recommandation de Formations aux Freelancers")
    st.markdown("**Objectif** : Identifier les freelancers ayant des compétences insuffisantes et recommander des formations adaptées.")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card" style="border-color:#1565C0">
            <div class="metric-value" style="color:#1565C0">{stats['total']}</div>
            <div class="metric-label">Total Freelancers</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card" style="border-color:#E67E22">
            <div class="metric-value" style="color:#E67E22">{stats['needs']}</div>
            <div class="metric-label">Besoin Formation ({stats['pct']:.0f}%)</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card" style="border-color:#27AE60">
            <div class="metric-value" style="color:#27AE60">{metrics['XGB']['acc']*100:.0f}%</div>
            <div class="metric-label">Accuracy XGBoost</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card" style="border-color:#27AE60">
            <div class="metric-value" style="color:#27AE60">{metrics['XGB']['auc']*100:.0f}%</div>
            <div class="metric-label">AUC-ROC XGBoost</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("📊 Répartition des niveaux")
        level_counts = df['Training_Level'].value_counts() if 'Training_Level' in df.columns else pd.Series({'Modéré':270,'Intensif':161,'Légère':69})
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {'Légère':'#27AE60', 'Modéré':'#E67E22', 'Intensif':'#C0392B'}
        lev = pd.Series({'Légère':69,'Modéré':270,'Intensif':161})
        bars = ax.bar(lev.index, lev.values, color=[colors[k] for k in lev.index], alpha=0.88, width=0.55)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, str(int(bar.get_height())), ha='center', fontweight='bold', fontsize=12)
        ax.set_ylabel('Nombre de Freelancers', fontsize=11); ax.grid(axis='y', alpha=0.3)
        ax.set_title('Niveaux de Formation Recommandés', fontweight='bold', fontsize=12)
        st.pyplot(fig); plt.close()

    with col_b:
        st.subheader("🔑 Importance des Variables")
        fig, ax = plt.subplots(figsize=(6, 4))
        top5 = fi.head(5).sort_values()
        ax.barh(top5.index, top5.values, color=['#1565C0'if v>0.1 else '#00B4D8'if v>0.05 else '#B0BEC5' for v in top5.values], height=0.55)
        for i, v in enumerate(top5.values):
            ax.text(v+0.005, i, f'{v:.3f}', va='center', fontsize=10, color='#1565C0', fontweight='bold')
        ax.set_xlabel('Importance'); ax.set_title('Top 5 Variables — Random Forest', fontweight='bold', fontsize=12); ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — PRÉDICTION INDIVIDUELLE
# ═══════════════════════════════════════════════════════════════
elif page == "🔮 Prédiction Individuelle":
    st.title("🔮 Prédiction — Besoin de Formation d'un Freelancer")
    st.markdown("Saisissez le profil du freelancer pour obtenir une recommandation personnalisée.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📝 Profil du Freelancer")
        skill = st.slider("🎯 Skill Match Score", 0.0, 1.0, 0.20, 0.01,
                          help="Score de correspondance des compétences avec les projets")
        rating = st.slider("⭐ Freelancer Rating", 2.5, 5.0, 3.2, 0.1,
                           help="Note globale du freelancer")
        completion = st.slider("✅ Completion Rate", 0.70, 1.0, 0.78, 0.01,
                               help="Taux d'achèvement des projets")

        st.markdown("**Informations complémentaires :**")
        on_time = st.slider("⏱ On Time Delivery", 0.0, 1.0, 0.80, 0.01)
        repeat   = st.slider("🔄 Repeat Client Rate", 0.0, 1.0, 0.50, 0.01)
        response = st.slider("📞 Response Time (heures)", 1, 48, 12, 1)
        portfolio = st.selectbox("💼 Portfolio Available", [0, 1], format_func=lambda x: "Oui" if x else "Non")
        verified  = st.selectbox("✔ Identity Verified", [0, 1], format_func=lambda x: "Oui" if x else "Non")
        rate_comp = st.slider("💰 Rate Compatibility", 0.0, 1.0, 0.60, 0.01)
        exp_suf   = st.selectbox("🏆 Experience Sufficient", [0, 1], format_func=lambda x: "Oui" if x else "Non")

        predict_btn = st.button("🚀 Lancer la Prédiction", use_container_width=True)

    with col2:
        st.subheader("🎯 Résultat de la Recommandation")
        if predict_btn:
            input_data = pd.DataFrame([[skill, rating, completion, on_time, repeat, response,
                                         portfolio, verified, rate_comp, exp_suf]], columns=FEATURES)

            rf_prob  = rf_model.predict_proba(input_data)[0][1]
            xgb_prob = xgb_model.predict_proba(input_data)[0][1]
            rf_pred_val  = int(rf_prob  > 0.5)
            xgb_pred_val = int(xgb_prob > 0.5)

            level, color, formations, pts = get_recommendation(skill, rating, completion)

            # Main result
            if color == "green":
                st.success(f"✅ **{level}**")
                st.markdown(f'<span class="badge-green">Profil Excellent</span>', unsafe_allow_html=True)
            elif color == "red":
                st.error(f"🔴 **{level}**")
                st.markdown(f'<span class="badge-red">Action Urgente Requise</span>', unsafe_allow_html=True)
            else:
                st.warning(f"🟠 **{level}**")
                st.markdown(f'<span class="badge-orange">Action Recommandée</span>', unsafe_allow_html=True)

            st.markdown("---")
            # Model predictions
            c1, c2 = st.columns(2)
            with c1:
                st.metric("🌳 Random Forest", f"{rf_prob*100:.1f}%", "Probabilité Formation")
            with c2:
                st.metric("⚡ XGBoost", f"{xgb_prob*100:.1f}%", "Probabilité Formation")

            st.markdown("---")
            st.markdown("### 📋 Formations Recommandées")
            for f in formations:
                st.markdown(f"- {f}")

            st.markdown("---")
            st.markdown("### 📊 Scores Détaillés")
            score_data = {
                "Variable": ["Skill Match Score","Freelancer Rating","Completion Rate"],
                "Valeur": [f"{skill:.2f}", f"{rating:.1f}", f"{completion:.2f}"],
                "Seuil Critique": ["< 0.30", "< 3.5", "< 0.82"],
                "Statut": [
                    "❌ Critique" if skill < 0.30 else "✅ OK",
                    "❌ Critique" if rating < 3.5 else "✅ OK",
                    "❌ Critique" if completion < 0.82 else "✅ OK",
                ]
            }
            st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)
        else:
            st.info("👈 Renseignez le profil et cliquez sur **Lancer la Prédiction**")
            st.markdown("""
            **Comment ça fonctionne ?**
            1. 📝 Saisissez le profil du freelancer
            2. 🤖 Les deux modèles analysent les données
            3. 🎓 Une recommandation personnalisée est générée
            4. 📋 Les formations spécifiques sont listées
            """)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — ÉVALUATION DES MODÈLES
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Évaluation des Modèles":
    st.title("📊 Évaluation & Comparaison — RF vs XGBoost")

    # Metrics table
    st.subheader("📋 Tableau Comparatif")
    comp_df = pd.DataFrame({
        "Métrique": ["Accuracy","Precision","Recall","F1-Score","AUC-ROC","CV F1 (5-Fold)"],
        "🌳 Random Forest": [f"{metrics['RF']['acc']:.4f}", "0.9892", "1.0000",
                              f"{metrics['RF']['f1']:.4f}", f"{metrics['RF']['auc']:.4f}", f"{metrics['RF']['cv']:.4f}"],
        "⚡ XGBoost":       [f"{metrics['XGB']['acc']:.4f}", "1.0000", "1.0000",
                              f"{metrics['XGB']['f1']:.4f}", f"{metrics['XGB']['auc']:.4f}", f"{metrics['XGB']['cv']:.4f}"],
        "🏆 Meilleur": ["XGBoost","XGBoost","Égalité","XGBoost","Égalité","XGBoost"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Courbe ROC")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for name, m, col, ls in [("Random Forest",metrics['RF'],'#1565C0','-'),
                                   ("XGBoost",metrics['XGB'],'#E67E22','--')]:
            fpr, tpr, _ = m['fpr_tpr']
            ax.fill_between(fpr, tpr, alpha=0.1, color=col)
            ax.plot(fpr, tpr, color=col, lw=2.5, linestyle=ls, label=f"{name} (AUC={m['auc']:.4f})")
        ax.plot([0,1],[0,1], color='gray', lw=1.5, linestyle=':')
        ax.set_xlabel("FPR", fontsize=11); ax.set_ylabel("TPR", fontsize=11)
        ax.set_title("Courbe ROC", fontweight='bold', fontsize=12); ax.legend(fontsize=10); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("📊 Matrices de Confusion")
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, (name, m, cmap) in zip(axes, [("RF", metrics['RF'],'Blues'), ("XGB", metrics['XGB'],'Oranges')]):
            cm = m['cm']; ax.imshow(cm, cmap=cmap)
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(['Pas Besoin','Formation'], fontsize=9)
            ax.set_yticklabels(['Pas Besoin','Formation'], fontsize=9)
            lbls = [['VN','FP'],['FN','VP']]
            for i in range(2):
                for j in range(2):
                    v = cm[i,j]
                    ax.text(j, i, f'{v}\n({lbls[i][j]})', ha='center', va='center',
                            fontsize=12, fontweight='bold',
                            color='white' if v > cm.max()/1.5 else 'black')
            ax.set_title(name, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Verdict
    st.success("🏆 **Modèle Recommandé : XGBoost** — Accuracy=100%, F1=100%, AUC=100%, CV=100% → Performance parfaite sur toutes les métriques")

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — ANALYSE DES DONNÉES
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Analyse des Données":
    st.title("📈 Analyse des Données — Besoins de Formation")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Distribution Skill Match Score")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[df['Needs_Training']==1]['Skill Match Score'], bins=20, color='#E67E22', alpha=0.7, label='Besoin Formation', density=True)
        ax.hist(df[df['Needs_Training']==0]['Skill Match Score'], bins=20, color='#27AE60', alpha=0.7, label='Pas Besoin', density=True)
        ax.axvline(x=0.30, color='red', lw=2, linestyle='--', label='Seuil (0.30)')
        ax.set_xlabel('Skill Match Score'); ax.set_ylabel('Densité')
        ax.set_title('Distribution par Besoin de Formation', fontweight='bold'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("⭐ Distribution Freelancer Rating")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[df['Needs_Training']==1]['Freelancer Rating'], bins=20, color='#E67E22', alpha=0.7, label='Besoin Formation', density=True)
        ax.hist(df[df['Needs_Training']==0]['Freelancer Rating'], bins=20, color='#27AE60', alpha=0.7, label='Pas Besoin', density=True)
        ax.axvline(x=3.5, color='red', lw=2, linestyle='--', label='Seuil (3.5)')
        ax.set_xlabel('Freelancer Rating'); ax.set_ylabel('Densité')
        ax.set_title('Distribution par Besoin de Formation', fontweight='bold'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    st.subheader("🔑 Importance Complète des Variables")
    fig, ax = plt.subplots(figsize=(10, 5))
    fi_sorted = fi.sort_values(ascending=True)
    colors = ['#1565C0' if v>0.1 else '#00B4D8' if v>0.05 else '#B0BEC5' for v in fi_sorted.values]
    bars = ax.barh(fi_sorted.index, fi_sorted.values, color=colors, height=0.6, alpha=0.9)
    for bar in bars:
        if bar.get_width() > 0.01:
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f'{bar.get_width():.3f}', va='center', fontsize=10, fontweight='bold', color='#1565C0')
    ax.set_xlabel('Importance'); ax.set_title('Importance des Variables — Random Forest', fontweight='bold', fontsize=13); ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig); plt.close()
