from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
import os
from datetime import datetime
import joblib
import numpy as np
from generar_mapa_interactivo import crear_grafico_interactivo_seguro


# Ruta a archivos
MODELO_PATH = "modelo_gmm.pkl"
SCALER_PATH = "scaler.pkl"
HISTORIAL_PATH = "historial_consultas.csv"
# Mapeo de cluster → línea profesional
MAPEO_CLUSTERS = {
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

# Configurar app Flask
app = Flask(__name__)
app.secret_key = 'clave_secreta_segura'  # Cambiar en producción

# Ruta a los datos de usuarios
USUARIOS_PATH = "usuarios.csv"

# === Ruta inicial: redirige a login ===
@app.route('/')
def index():
    return redirect(url_for('login'))

# === Página de login ===
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        id_ingresado = request.form['id'].strip()
        contrasena_ingresada = request.form['password'].strip()

        df = pd.read_csv(USUARIOS_PATH)
        usuario = df[df['ID'].astype(str) == id_ingresado]

        if not usuario.empty:
            contrasena_real = str(usuario.iloc[0]['Contraseña']).strip()
            if contrasena_real == contrasena_ingresada:
                session['id_usuario'] = int(usuario.iloc[0]['ID'])
                return redirect(url_for('perfil'))


        return render_template('login.html', error="ID o contraseña incorrectos")

    return render_template('login.html')


# === Página de perfil ===
@app.route('/perfil')
def perfil():
    if 'id_usuario' not in session:
        return redirect(url_for('login'))

    # Cargar usuario
    df = pd.read_csv(USUARIOS_PATH)
    usuario = df[df['ID'] == session['id_usuario']].iloc[0]

    # Calificaciones por curso
    columnas_notas = [col for col in df.columns if '°' in col]
    notas_usuario = {col: int(usuario[col]) for col in columnas_notas}

    # Historial
    if os.path.exists(HISTORIAL_PATH):
        historial_df = pd.read_csv(HISTORIAL_PATH)
        historial_usuario = historial_df[historial_df['ID'] == session['id_usuario']].sort_values(by="Fecha", ascending=False).to_dict('records')
        sugerencia_principal = historial_usuario[0]["Sugerencia"] if historial_usuario else "Sin consultas"
    else:
        historial_usuario = []
        sugerencia_principal = "Sin consultas"

    # 🔹 Calcular promedios por área para el radar chart
    promedios_area = calcular_promedios_por_area(usuario)

    return render_template("perfil.html",
        nombre=usuario['Nombre'],
        linea_actual=sugerencia_principal,
        calificaciones=notas_usuario,
        historial=historial_usuario,
        promedios=promedios_area  # 🔹 Nuevo
    )
    

@app.route('/consulta', methods=['POST'])
def consulta():
    if 'id_usuario' not in session:
        return redirect(url_for('login'))

    # Cargar usuario
    df = pd.read_csv(USUARIOS_PATH)
    usuario = df[df['ID'] == session['id_usuario']].iloc[0]

    # Extraer calificaciones
    columnas_notas = [col for col in df.columns if '°' in col]
    notas = np.array([usuario[col] for col in columnas_notas]).reshape(1, -1)

    # Cargar scaler y modelo
    scaler = joblib.load(SCALER_PATH)
    modelo = joblib.load(MODELO_PATH)

    notas_escaladas = scaler.transform(notas)

    # Predecir probabilidades
    probs = modelo.predict_proba(notas_escaladas)[0]
    top_indices = np.argsort(probs)[::-1][:3]  # Top 3 clusters

    # Mapear sugerencias
    sugerencias = [(MAPEO_CLUSTERS.get(i, "Desconocido"), round(probs[i] * 100, 2)) for i in top_indices]
    cluster_predicho = int(top_indices[0])
    sugerencia_principal = sugerencias[0][0]

    # Guardar en historial
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nuevo_registro = pd.DataFrame([{
        "ID": session['id_usuario'],
        "Fecha": fecha,
        "Cluster": cluster_predicho,
        "Sugerencia": sugerencia_principal
    }])
    if os.path.exists(HISTORIAL_PATH):
        historial = pd.read_csv(HISTORIAL_PATH)
        historial = pd.concat([historial, nuevo_registro], ignore_index=True)
    else:
        historial = nuevo_registro
    historial.to_csv(HISTORIAL_PATH, index=False)

    # Mostrar resultado
    distribucion_local = crear_grafico_interactivo_seguro(session['id_usuario'])
    return render_template("consulta.html",
        nombre=usuario['Nombre'],
        sugerencia=sugerencia_principal,
        cluster=cluster_predicho,
        fecha=fecha,
        distribucion=distribucion_local
    )

def calcular_promedios_por_area(usuario):
    areas = ["Matemática", "Comunicación", "Ciencia", "Arte", "Personal Social"]
    promedios = {}
    for area in areas:
        notas_area = [usuario[col] for col in usuario.index if area in col]
        promedio = sum(notas_area) / len(notas_area) if notas_area else 0
        promedios[area] = round(promedio, 2)
    return promedios

# === Logout ===
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# === Iniciar servidor local ===
if __name__ == '__main__':
    app.run(debug=True)
