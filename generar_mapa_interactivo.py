import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

def crear_grafico_interactivo_seguro(id_estudiante: int):
    try:
        # === Cargar datos ===
        df = pd.read_excel("datos_simulados_orientacion_vocacional.xlsx")
        columnas_notas = [col for col in df.columns if '°' in col]
        X = df[columnas_notas]
        y = df['Linea']
        ids = df['ID']

        # === PCA ===
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # === DataFrame completo (interno)
        pca_df = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'Linea': y,
            'ID': ids
        })

        # === Obtener coordenadas del estudiante actual
        estudiante = pca_df[pca_df['ID'].astype(str) == str(id_estudiante)]
        if estudiante.empty:
            print(f"[!] ID {id_estudiante} no encontrado")
            return {}

        x_ = estudiante.iloc[0]['PCA1']
        y_ = estudiante.iloc[0]['PCA2']
        linea_ = estudiante.iloc[0]['Linea']

        # === Distribución vocacional en el área
        r = 0.7  # mismo radio del círculo
        pca_df['distancia'] = np.sqrt((pca_df['PCA1'] - x_)**2 + (pca_df['PCA2'] - y_)**2)
        dentro = pca_df[pca_df['distancia'] <= r]
        conteo = Counter(dentro['Linea'])
        total = sum(conteo.values())
        distribucion_local = {
            linea: round((cantidad / total) * 100, 2)
            for linea, cantidad in conteo.items()
        }
        distribucion_local = dict(sorted(distribucion_local.items(), key=lambda x: x[1], reverse=True))

        # === Eliminar columnas sensibles para graficar
        pca_df_safe = pca_df.drop(columns=["ID", "distancia"])

        # === Crear gráfico base sin ID
        fig = px.scatter(
            pca_df_safe,
            x="PCA1",
            y="PCA2",
            color="Linea",
            hover_data={"Linea": True, "PCA1": False, "PCA2": False},
            title="Mapa PCA interactivo de orientación vocacional"
        )

        # === Agregar círculo transparente
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=x_ - r, x1=x_ + r,
            y0=y_ - r, y1=y_ + r,
            line=dict(color="red", width=2),
            fillcolor="rgba(255, 0, 0, 0.2)",
            layer="above"
        )

        # === Agregar flecha con texto
        fig.add_annotation(
            x=x_,
            y=y_,
            ax=x_ - 2.5,
            ay=y_ + 2.5,
            xref="x", yref="y",
            axref="x", ayref="y",
            text="Aquí estás tú",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='red',
            font=dict(color='red', size=12)
        )

        # === Centrar el gráfico en el estudiante
        fig.update_layout(
            xaxis=dict(range=[x_ - 4, x_ + 4]),
            yaxis=dict(range=[y_ - 4, y_ + 4]),
            hovermode="closest",
            legend_title="Línea profesional"
        )

        # === Guardar como HTML
        fig.write_html("static/mapa_pca_interactivo.html")
        print(f"✅ Gráfico generado para ID {id_estudiante}")

        # === Retornar distribución
        return distribucion_local

    except Exception as e:
        print(f"[ERROR al generar gráfico PCA]: {e}")
        return {}
