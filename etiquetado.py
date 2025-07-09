import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar archivo con resultados
df = pd.read_excel("cluster_resultados.xlsx")  # Este archivo debe venir desde ml.py

# 2. Crear tabla cruzada de conteo: Línea Profesional (real) vs Cluster GMM
tabla = pd.crosstab(df["Cluster"], df["Linea"])

# 3. Mostrar tabla por consola
print("\n=== Distribución de líneas profesionales dentro de cada cluster ===")
print(tabla)

# 4. Graficar como heatmap para visualización clara
plt.figure(figsize=(14, 7))
sns.heatmap(tabla, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Frecuencia de líneas profesionales por cluster GMM")
plt.xlabel("Línea Profesional (real)")
plt.ylabel("Cluster asignado por GMM")
plt.tight_layout()
plt.show()

# 5. Sugerir etiquetas por cluster: la línea profesional más frecuente
print("\n=== Sugerencias de etiqueta para cada cluster ===")
sugerencias = {}
for cluster in tabla.index:
    dominante = tabla.loc[cluster].idxmax()
    porcentaje = tabla.loc[cluster].max() / tabla.loc[cluster].sum()
    sugerencias[cluster] = dominante
    print(f"Cluster {cluster}: {dominante} ({porcentaje:.1%} del cluster)")

# 6. Guardar tabla cruzada con etiquetas sugeridas (opcional)
tabla["Sugerencia_etiqueta"] = tabla.idxmax(axis=1)
tabla.to_excel("analisis_clusters_etiquetados.xlsx")
print("\nArchivo guardado: analisis_clusters_etiquetados.xlsx")
