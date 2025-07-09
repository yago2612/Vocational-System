import pandas as pd
import random

# Cargar el archivo de calificaciones simuladas
df = pd.read_excel("datos_simulados_orientacion_vocacional.xlsx")

# Crear nombres ficticios
nombres = ["Juan", "María", "Luis", "Ana", "Pedro", "Lucía", "Diego", "Elena", "Carlos", "Sofía"]
apellidos = ["Pérez", "Gómez", "Rodríguez", "Fernández", "Ramírez", "López", "Castro", "Vargas", "Flores", "Torres"]

usuarios = []
for idx, row in df.iterrows():
    nombre = random.choice(nombres) + " " + random.choice(apellidos)
    contrasena = "123456"  # Contraseña por defecto
    usuario = {
        "ID": row["ID"],
        "Nombre": nombre,
        "Contraseña": contrasena,
        "Linea": row["Linea"]
    }
    # Agregar todas las calificaciones
    for col in df.columns:
        if col not in ["ID", "Linea"]:
            usuario[col] = row[col]
    usuarios.append(usuario)

# Guardar como CSV
df_usuarios = pd.DataFrame(usuarios)
df_usuarios.to_csv("usuarios.csv", index=False)
print("Archivo generado: usuarios.csv")
