import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

# === Cargar datos
df = pd.read_excel("datos_simulados_orientacion_vocacional.xlsx")
columnas_notas = [col for col in df.columns if '°' in col]
X = df[columnas_notas]
y_real = df['Linea']

# === Cargar scaler y modelo
scaler = joblib.load("scaler.pkl")
modelo = joblib.load("modelo_gmm.pkl")
X_scaled = scaler.transform(X)

# === Separar datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_real, test_size=0.15, random_state=42)

# === Predicción
train_pred = modelo.predict(X_train)
test_pred = modelo.predict(X_test)
full_pred = modelo.predict(X_scaled)

# === Métricas
silhouette_train = silhouette_score(X_train, train_pred)
silhouette_test = silhouette_score(X_test, test_pred)
ari_total = adjusted_rand_score(y_real, full_pred)

print("=== Evaluación del Modelo ===")
print(f"Silhouette Score (Train): {silhouette_train:.4f}")
print(f"Silhouette Score (Test): {silhouette_test:.4f}")
print(f"Adjusted Rand Index (Total): {ari_total:.4f}")
