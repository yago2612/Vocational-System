import pandas as pd
import numpy as np
import random

# Semilla para reproducibilidad
random.seed(42)
np.random.seed(42)

# Cursos por estudiante por grado
cursos = ["Matemática", "Comunicación", "Ciencia y Ambiente", "Personal Social", "Arte"]
grados = ["1°", "2°", "3°", "4°", "5°"]

# Distribución por línea profesional
lineas_profesionales = {
    "Agricultura, pesca y veterinaria": 111,
    "Arte y humanidades": 298,
    "Ciencias administrativas y derecho": 3315,
    "Ciencias naturales, matemáticas y estadística": 248,
    "Ciencias sociales, periodismo e información": 1494,
    "Educación": 261,
    "Ingeniería, industria y construcción": 2086,
    "Salud y bienestar": 1250,
    "Servicios": 182,
    "Tecnología de la información y la comunicación": 692
}

# Mapas de patrones (media, desviación) por grupo
# Puedes ajustar estos valores luego
mapa_patrones = {
    "Ingeniería, industria y construcción": {
        "Matemática":       [17.0, 1.0],
        "Comunicación":     [13.5, 1.5],
        "Ciencia y Ambiente": [16.5, 1.0],
        "Personal Social":  [12.0, 1.5],
        "Arte":             [12.5, 2.0]
    },
    "Arte y humanidades": {
        "Matemática":       [11.5, 1.5],
        "Comunicación":     [16.0, 1.2],
        "Ciencia y Ambiente": [12.0, 1.0],
        "Personal Social":  [14.0, 1.0],
        "Arte":             [17.0, 1.0]
    },
    "Ciencias naturales, matemáticas y estadística": {
        "Matemática":       [17.0, 1.0],
        "Comunicación":     [13.0, 1.0],
        "Ciencia y Ambiente": [17.0, 0.8],
        "Personal Social":  [13.5, 1.2],
        "Arte":             [12.5, 1.0]
    },
    "Ciencias sociales, periodismo e información": {
        "Matemática":       [13.5, 1.5],
        "Comunicación":     [16.0, 1.0],
        "Ciencia y Ambiente": [14.0, 1.2],
        "Personal Social":  [16.0, 1.0],
        "Arte":             [14.0, 1.0]
    },
    "Ciencias administrativas y derecho": {
        "Matemática":       [14.5, 1.0],
        "Comunicación":     [15.0, 1.0],
        "Ciencia y Ambiente": [13.0, 1.0],
        "Personal Social":  [15.5, 1.2],
        "Arte":             [13.0, 1.0]
    },
    "Salud y bienestar": {
        "Matemática":       [15.5, 1.0],
        "Comunicación":     [14.0, 1.2],
        "Ciencia y Ambiente": [16.0, 1.0],
        "Personal Social":  [14.5, 1.0],
        "Arte":             [12.5, 1.2]
    },
    "Educación": {
        "Matemática":       [13.5, 1.0],
        "Comunicación":     [15.5, 1.0],
        "Ciencia y Ambiente": [14.0, 1.0],
        "Personal Social":  [16.0, 1.0],
        "Arte":             [14.0, 1.0]
    },
    "Servicios": {
        "Matemática":       [13.0, 1.0],
        "Comunicación":     [14.0, 1.2],
        "Ciencia y Ambiente": [12.0, 1.0],
        "Personal Social":  [14.0, 1.0],
        "Arte":             [14.0, 1.0]
    },
    "Agricultura, pesca y veterinaria": {
        "Matemática":       [13.5, 1.2],
        "Comunicación":     [12.5, 1.0],
        "Ciencia y Ambiente": [14.0, 1.2],
        "Personal Social":  [13.5, 1.0],
        "Arte":             [12.0, 1.0]
    },
    "Tecnología de la información y la comunicación": {
        "Matemática":       [17.0, 1.0],
        "Comunicación":     [12.5, 1.0],
        "Ciencia y Ambiente": [15.0, 1.0],
        "Personal Social":  [11.5, 1.0],
        "Arte":             [12.0, 1.0]
    }
}

def generar_notas(media, std, caida_3ro=False):
    notas = []
    for i in range(5):
        m = media
        s = std
        if i == 2 and caida_3ro:
            m -= 1  # Caída realista en 3º
        nota = np.clip(np.random.normal(m, s), 10, 20)
        notas.append(round(nota, 1))
    return notas

# Simular datos
data = []

id_counter = 1
for linea, cantidad in lineas_profesionales.items():
    patron = mapa_patrones[linea]
    for _ in range(cantidad):
        row = {"ID": id_counter, "Linea": linea}
        for curso in cursos:
            caida = curso == "Personal Social" and linea in ["Ingeniería, industria y construcción", "Tecnología de la información y la comunicación", "Ciencias naturales, matemáticas y estadística"]
            notas = generar_notas(patron[curso][0], patron[curso][1], caida_3ro=caida)
            for i, grado in enumerate(grados):
                row[f"{curso}_{grado}"] = notas[i]
        data.append(row)
        id_counter += 1

# Exportar a Excel
df = pd.DataFrame(data)
df.to_excel("datos_simulados_orientacion_vocacional.xlsx", index=False)
print("Archivo Excel generado: datos_simulados_orientacion_vocacional.xlsx")
