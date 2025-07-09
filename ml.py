import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar datos simulados
df = pd.read_excel("datos_simulados_orientacion_vocacional.xlsx")

# 2. Separar variables predictoras (X) y etiqueta real (y)
X = df.drop(columns=["ID", "Linea"])
y = df["Linea"]  # Solo para comparar después

# 3. Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42
)

# 5. Entrenar GMM
n_clusters = 10  # Porque sabemos que simulamos 10 grupos
gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
gmm.fit(X_train)

# 6. Predecir clusters
train_preds = gmm.predict(X_train)
test_preds = gmm.predict(X_test)

# 7. Evaluación del modelo
sil_train = silhouette_score(X_train, train_preds)
sil_test = silhouette_score(X_test, test_preds)
print(f"Silhouette Score (Train): {sil_train:.4f}")
print(f"Silhouette Score (Test): {sil_test:.4f}")

# Comparación con etiquetas reales (solo para evaluación final)
# Esta métrica NO usa las etiquetas durante el entrenamiento
true_test_labels = y_test.values
ari = adjusted_rand_score(true_test_labels, test_preds)
print(f"Adjusted Rand Index (con etiquetas reales, solo para validación): {ari:.4f}")

# 8. Visualización con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=test_preds,
    palette='tab10',
    legend="full",
    alpha=0.8
)

# Asignar clusters a todos los datos (usando X_scaled completo)
clusters_finales = gmm.predict(X_scaled)

# Crear DataFrame de resultados con ID, línea real y cluster asignado
df_resultado = df[["ID", "Linea"]].copy()
df_resultado["Cluster"] = clusters_finales

# Guardar a Excel para uso en etiquetado
df_resultado.to_excel("cluster_resultados.xlsx", index=False)
print("Archivo generado: cluster_resultados.xlsx")

# Guardar el modelo GMM entrenado
joblib.dump(gmm, "modelo_gmm.pkl")
print("Modelo GMM guardado como modelo_gmm.pkl")
# Guardar el scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler guardado como scaler.pkl")


plt.title("Clusters GMM sobre Datos de Calificaciones (Test Set, Reducción PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Cluster GMM")
plt.grid(True)
plt.tight_layout()
plt.show()
