import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------- Configuration ---------------------------
st.set_page_config(page_title="Logiciel PV ML", layout="wide")
st.title("Logiciel de prédiction de la Puissance PV avec Machine Learning")

# --------------------------- Sidebar ---------------------------
menu = ["Données", "Modèles", "Graphiques", "Comparaison", "Prédiction directe"]
choice = st.sidebar.selectbox("Menu", menu)

# --------------------------- Upload fichier ---------------------------
file = st.file_uploader("Importer le fichier Excel", type=["xlsx"])

if file is not None:
    df = pd.read_excel(file)
    df = df.dropna()
    colonnes = [
        "Rayonnement global plan PV (W/m2)",
        "Température cellule PV (C°)",
        "Température generateur PV (C°)",
        "Température Batterie (C°)",
        "Température Ambiante (C°)",
        "Consommation sortie Ond.(W)",
        "Puissance PV"
    ]
    scaler = MinMaxScaler()
    df[colonnes] = scaler.fit_transform(df[colonnes])

    # Variables explicatives et cible
    X = df[[
        "Rayonnement global plan PV (W/m2)",
        "Température cellule PV (C°)",
        "Température generateur PV (C°)",
        "Température Ambiante (C°)"
    ]]
    y = df["Puissance PV"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Création et entraînement des modèles
    model_lr = LinearRegression()
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_dt = DecisionTreeRegressor(random_state=42)

    model_lr.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)
    model_dt.fit(X_train, y_train)

    # Prédictions sur test
    y_pred_lr = model_lr.predict(X_test)
    y_pred_rf = model_rf.predict(X_test)
    y_pred_dt = model_dt.predict(X_test)

    # Metrics
    metrics_df = pd.DataFrame({
        "Modèle": ["Régression Linéaire", "Random Forest", "Decision Tree"],
        "MAE": [
            mean_absolute_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_rf),
            mean_absolute_error(y_test, y_pred_dt)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            np.sqrt(mean_squared_error(y_test, y_pred_dt))
        ],
        "R²": [
            r2_score(y_test, y_pred_lr),
            r2_score(y_test, y_pred_rf),
            r2_score(y_test, y_pred_dt)
        ]
    })

    # --------------------------- Pages ---------------------------
    if choice == "Données":
        st.subheader("Aperçu des données")
        st.write(df.head())
        st.write("Dimensions :", df.shape)

        st.subheader("Matrice de corrélation")
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

    elif choice == "Modèles":
        st.subheader("Performance des modèles")
        st.write(metrics_df)

    elif choice == "Graphiques":
        st.subheader("Graphiques par modèle")

        # Régression Linéaire
        st.markdown("### Régression Linéaire")
        fig_lr, ax_lr = plt.subplots()
        ax_lr.scatter(y_test, y_pred_lr, color='blue', alpha=0.6, label='Prédiction')
        ax_lr.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Réel')
        ax_lr.set_xlabel("Puissance réelle")
        ax_lr.set_ylabel("Puissance prédite")
        ax_lr.legend()
        st.pyplot(fig_lr)

        # Random Forest
        st.markdown("### Random Forest")
        fig_rf, ax_rf = plt.subplots()
        ax_rf.scatter(y_test, y_pred_rf, color='green', alpha=0.6, label='Prédiction')
        ax_rf.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Réel')
        ax_rf.set_xlabel("Puissance réelle")
        ax_rf.set_ylabel("Puissance prédite")
        ax_rf.legend()
        st.pyplot(fig_rf)

        # Decision Tree
        st.markdown("### Decision Tree")
        fig_dt, ax_dt = plt.subplots()
        ax_dt.scatter(y_test, y_pred_dt, color='purple', alpha=0.6, label='Prédiction')
        ax_dt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Réel')
        ax_dt.set_xlabel("Puissance réelle")
        ax_dt.set_ylabel("Puissance prédite")
        ax_dt.legend()
        st.pyplot(fig_dt)

    elif choice == "Comparaison":
        st.subheader("Comparaison des modèles")
        fig_comp, ax_comp = plt.subplots()
        ax_comp.scatter(y_test, y_pred_lr, color='blue', alpha=0.6, label='Régression Linéaire')
        ax_comp.scatter(y_test, y_pred_rf, color='green', alpha=0.6, label='Random Forest')
        ax_comp.scatter(y_test, y_pred_dt, color='purple', alpha=0.6, label='Decision Tree')
        ax_comp.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Réel')
        ax_comp.set_xlabel("Puissance réelle")
        ax_comp.set_ylabel("Puissance prédite")
        ax_comp.legend()
        st.pyplot(fig_comp)

    elif choice == "Prédiction directe":
        st.subheader("Prédiction interactive de la puissance PV")
        st.write("Entrez les valeurs de Rayonnement et Températures (normalisées entre 0 et 1) :")

        # Sliders pour les valeurs
        rayonnement = st.slider("Rayonnement global plan PV (W/m2)", 0.0, 1.0, 0.5)
        temp_cellule = st.slider("Température cellule PV (C°)", 0.0, 1.0, 0.5)
        temp_generateur = st.slider("Température générateur PV (C°)", 0.0, 1.0, 0.5)
        temp_ambiante = st.slider("Température Ambiante (C°)", 0.0, 1.0, 0.5)

        if st.button("Prédire"):
            input_data = np.array([[rayonnement, temp_cellule, temp_generateur, temp_ambiante]])
            pred_lr = model_lr.predict(input_data)[0]
            pred_rf = model_rf.predict(input_data)[0]
            pred_dt = model_dt.predict(input_data)[0]

            st.markdown(f"**Régression Linéaire :** {pred_lr:.4f}")
            st.markdown(f"**Random Forest :** {pred_rf:.4f}")
            st.markdown(f"**Decision Tree :** {pred_dt:.4f}")

            # Graphique instantané
            fig_pred, ax_pred = plt.subplots()
            models = ["Linéaire", "Random Forest", "Decision Tree"]
            predictions = [pred_lr, pred_rf, pred_dt]
            ax_pred.bar(models, predictions, color=['blue', 'green', 'purple'])
            ax_pred.set_ylabel("Puissance PV prédite")
            ax_pred.set_title("Prédictions par modèle")
            st.pyplot(fig_pred)

            # Export CSV
            pred_df = pd.DataFrame({
                "Modèle": models,
                "Puissance PV prédite": predictions
            })
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("Exporter les prédictions CSV", data=csv, file_name='predictions.csv', mime='text/csv')