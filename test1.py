import pandas as pd
import numpy as np
import joblib

# === 1. Cargar modelo y scaler previamente guardados ===
gmm = joblib.load("modelo_gmm.pkl")
scaler = joblib.load("scaler.pkl")

# === 2. Definir calificaciones simuladas del estudiante ===
# NOTA: deben tener el mismo orden y estructura que X_scaled (25 columnas)
# Ejemplo: Mat_1°, Mat_2°, ..., Arte_5°
# Sustituye los valores según pruebas reales
datos_usuario = {
    "Matemática_1°": 14, "Matemática_2°": 13, "Matemática_3°": 15, "Matemática_4°": 16, "Matemática_5°": 17,
    "Comunicación_1°": 13, "Comunicación_2°": 14, "Comunicación_3°": 14, "Comunicación_4°": 15, "Comunicación_5°": 15,
    "Ciencia y Ambiente_1°": 13, "Ciencia y Ambiente_2°": 13, "Ciencia y Ambiente_3°": 14, "Ciencia y Ambiente_4°": 16, "Ciencia y Ambiente_5°": 17,
    "Personal Social_1°": 12, "Personal Social_2°": 12, "Personal Social_3°": 11, "Personal Social_4°": 13, "Personal Social_5°": 13,
    "Arte_1°": 12, "Arte_2°": 13, "Arte_3°": 14, "Arte_4°": 14, "Arte_5°": 14
}

# Convertir a DataFrame y escalar
df_usuario = pd.DataFrame([datos_usuario])
df_usuario_scaled = scaler.transform(df_usuario)

# === 3. Predecir cluster ===
cluster_predicho = gmm.predict(df_usuario_scaled)[0]
print(f"Cluster asignado por el modelo: {cluster_predicho}")

# === 4. Mapeo de cluster → línea profesional (según etiquetado.py) ===
mapeo_clusters = {
    0: "Salud y bienestar",
    1: "Ciencias sociales, periodismo e información",
    2: "Ingeniería, industria y construcción",
    3: "Tecnología de la información y la comunicación",
    4: "Servicios",
    5: "Arte y humanidades",
    6: "Arte y humanidades",
    7: "Ciencias administrativas y derecho",
    8: "Ciencias sociales, periodismo e información",
    9: "Ciencias administrativas y derecho"
}

sugerencia = mapeo_clusters.get(cluster_predicho, "Desconocido")
print(f"Sugerencia vocacional: {sugerencia}")
