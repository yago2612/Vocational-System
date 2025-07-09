import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# === Cargar datos ===
df = pd.read_excel("datos_simulados_orientacion_vocacional.xlsx")

# Calificaciones
columnas_notas = [col for col in df.columns if '°' in col]
X = df[columnas_notas]
y = df['Linea']
ids = df['ID']

# === Estandarizar ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PCA a 2 componentes ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Crear DataFrame con componentes y línea profesional
pca_df = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'Linea': y,
    'ID': ids
})

# === Graficar ===
plt.figure(figsize=(12, 8))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Linea', palette='tab10', alpha=0.7, edgecolor=None)
# Agregar etiquetas de línea profesional en el centro del grupo
for linea in pca_df['Linea'].unique():
    subset = pca_df[pca_df['Linea'] == linea]
    x_mean = subset['PCA1'].mean()
    y_mean = subset['PCA2'].mean()
    plt.text(x_mean, y_mean, linea, fontsize=9, weight='bold', ha='center')

plt.title("Mapa PCA de perfiles vocacionales por línea profesional")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Guardar imagen
plt.savefig("mapa_pca_lineas.png", dpi=300)
print("Gráfico guardado como mapa_pca_lineas.png")
