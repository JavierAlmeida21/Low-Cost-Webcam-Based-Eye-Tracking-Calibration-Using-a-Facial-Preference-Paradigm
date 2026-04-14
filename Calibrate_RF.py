import cv2
import numpy as np
import time
from collections import deque
from gaze_tracking import GazeTracking
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pyautogui
import os
from tkinter import *
import tkinter as tk
import sys
from sklearn.metrics import mean_squared_error
import math
from PIL import Image, ImageDraw, ImageFont, Image
import ctypes


# Inicializar gaze tracking
gaze = GazeTracking()

# --------------------------------------------
# Cambiar la ruta al directorio actual de manera automática
# Manejo universal de rutas: .py o .exe
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS  # solo para archivos empaquetados
    output_path = os.path.dirname(sys.executable)  # donde está el .exe
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = base_path  # para desarrollo, salida en misma carpeta

print(f"Ruta base detectada: {base_path}")

########### ------------------------------------- ####################################

def mostrar_instrucciones(ruta_carpeta):
    ctypes.windll.user32.ShowCursor(False)
    # Obtiene lista de imágenes soportadas en orden alfabético
    imagenes = sorted([
        os.path.join(ruta_carpeta, f)
        for f in os.listdir(ruta_carpeta)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    if not imagenes:
        print("No se encontraron imágenes de instrucciones en la carpeta.")
        return

    screen_width, screen_height = pyautogui.size()

    for img_path in imagenes:
        img = cv2.imread(img_path)

        if img is None:
            print(f"No se pudo leer la imagen: {img_path}")
            continue

        # Escalar la imagen a pantalla completa manteniendo proporciones
        img_ratio = img.shape[1] / img.shape[0]
        screen_ratio = screen_width / screen_height

        if img_ratio > screen_ratio:
            new_width = screen_width
            new_height = int(screen_width / img_ratio)
        else:
            new_height = screen_height
            new_width = int(screen_height * img_ratio)

        img_resized = cv2.resize(img, (new_width, new_height))

        # Crear lienzo negro de pantalla completa y centrar la imagen
        canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        x_offset = (screen_width - new_width) // 2
        y_offset = (screen_height - new_height) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img_resized

        cv2.namedWindow("Instrucciones", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Instrucciones", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            cv2.imshow("Instrucciones", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Barra espaciadora
                break
            elif key == 27:  # ESC para salir de emergencia
                cv2.destroyAllWindows()
                sys.exit(0)

    cv2.destroyAllWindows()

def verificacion_pre_calibracion(cap):
    # Configuración
    ctypes.windll.user32.ShowCursor(False)
    emoji_path = os.path.join(base_path, "Proyecto_Eye_Tracking", "Proyecto", "Imagenes_Interfaz", "emoji.png")
    fuente = ImageFont.truetype("arial.ttf", 28)

    # Validar carga de emoji
    if not os.path.isfile(emoji_path):
        raise FileNotFoundError(f"No se encontró la imagen: {emoji_path}")
    emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    emoji_img = cv2.resize(emoji_img, (40, 40))

    cv2.namedWindow("Verificación", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Verificación", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ret, frame = cap.read()
    if not ret:
        print("No se pudo iniciar la cámara para verificación.")
        cap.release()
        return
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    screen_width, screen_height = pyautogui.size()
    OVAL_CENTER = (frame_width // 2, frame_height // 2)
    OVAL_SIZE = (90, 150)

    eje_x_vals, eje_y_vals = [], []

    feedback = "Ubica tu rostro correctamente"
    perfect_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        visual_frame = np.zeros_like(frame)
        cv2.ellipse(visual_frame, OVAL_CENTER, OVAL_SIZE, 0, 0, 360, (0, 255, 0), 2)

        processed = cv2.equalizeHist(gray)
        processed = cv2.GaussianBlur(processed, (5, 5), 0)
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        gaze.refresh(processed_bgr)
        pupil_coords = gaze.pupil_right_coords()

        feedback = "Ubica tu rostro correctamente"

        if pupil_coords:
            eje_x_vals.append(pupil_coords[0])
            eje_y_vals.append(pupil_coords[1])
            min_x, max_x = min(eje_x_vals), max(eje_x_vals)
            min_y, max_y = min(eje_y_vals), max(eje_y_vals)
            rango_x = max_x - min_x if max_x - min_x != 0 else 1
            rango_y = max_y - min_y if max_y - min_y != 0 else 1
            normalizado_x = (pupil_coords[0] - min_x) / rango_x
            normalizado_y = (pupil_coords[1] - min_y) / rango_y

            x_escalado = int(normalizado_x * screen_width)
            y_escalado = int(normalizado_y * screen_height)
            x_frame = int((x_escalado / screen_width) * frame_width)
            y_frame = int((y_escalado / screen_height) * frame_height)

            dx = x_frame - OVAL_CENTER[0]
            dy = y_frame - OVAL_CENTER[1]
            dentro_del_ovalo = (dx**2 / OVAL_SIZE[0]**2 + dy**2 / OVAL_SIZE[1]**2) <= 1

            if dentro_del_ovalo:
                feedback = "Ubicación perfecta"
                if perfect_start_time is None:
                    perfect_start_time = cv2.getTickCount()
                else:
                    elapsed_time = (cv2.getTickCount() - perfect_start_time) / cv2.getTickFrequency()
                    if elapsed_time >= 3.0:
                        break
            else:
                feedback = "Ajusta tu posición"
                perfect_start_time = None

            # Dibujar emoji con transparencia
            x_offset = x_frame - emoji_img.shape[1] // 2
            y_offset = y_frame - emoji_img.shape[0] // 2
            if 0 <= x_offset <= frame_width - emoji_img.shape[1] and 0 <= y_offset <= frame_height - emoji_img.shape[0]:
                for c in range(3):  # canales BGR
                    alpha = emoji_img[:, :, 3] / 255.0
                    visual_frame[y_offset:y_offset+emoji_img.shape[0], x_offset:x_offset+emoji_img.shape[1], c] = (
                        alpha * emoji_img[:, :, c] +
                        (1 - alpha) * visual_frame[y_offset:y_offset+emoji_img.shape[0], x_offset:x_offset+emoji_img.shape[1], c]
                    )
        else:
            perfect_start_time = None

        # Mostrar texto con PIL para mejor renderizado
        frame_rgb = cv2.cvtColor(visual_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_pil)
        color = (0, 200, 0) if feedback == "Ubicación perfecta" else (255, 0, 0)
        draw.text((30, 40), feedback, font=fuente, fill=color)
        final_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("Verificación", final_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == 32:
            break

    cv2.destroyAllWindows()


def mostrar_transicion(mensaje):
    trans_root = tk.Tk()
    trans_root.attributes('-fullscreen', True)
    trans_root.configure(bg='black')

    label = tk.Label(trans_root, text=mensaje, font=("Century Ghotic", 28), fg='white', bg='black')
    label.pack(expand=True)

    trans_root.update()
    trans_root.after(2000, trans_root.destroy)  # Muestra 2.5 segundos
    trans_root.mainloop()




def comenzar_deteccion(codigo):

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  #Dejar 0 por defecto, la mayoría de computadores tienen esta dirección para su cámara, a menos que haya una externa.

    verificacion_pre_calibracion(cap)
    mostrar_transicion("Cargando calibración...")
    gaze = GazeTracking()  #Setear el gaze tracking para evitar conflictos con la calibración
    screen_width, screen_height = pyautogui.size()

    # Ventana pantalla completa
    window_name = "Gaze Calibrator"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Generar puntos 3x3 (proporciones)
    proportions = [0.1, 0.5, 0.9]
    calibration_points = [(int(screen_width * x), int(screen_height * y))for y in proportions for x in proportions]
    calibration_data = []

    ctypes.windll.user32.ShowCursor(False)
    def draw_calibration_point(point):
        bg = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        cv2.circle(bg, point, 25, (0, 0, 255), -1)
        return bg

    print("Iniciando calibración con 9 puntos...")

    for idx, point in enumerate(calibration_points):
        samples = []
        start_time = time.time()

        while time.time() - start_time < 2.0:
            ret, frame = cap.read()
            if not ret:
                continue
            gaze.refresh(frame)
            hr = gaze.horizontal_ratio()
            vr = gaze.vertical_ratio()
            if hr is not None and vr is not None:
                samples.append((hr, vr))
            screen = draw_calibration_point(point)       
            cv2.imshow(window_name, screen)
            cv2.waitKey(1)

        if not samples:
            continue

        # Filtrar muestras fuera de ±0.05 del promedio para hr y vr
        mean_hr = np.mean([s[0] for s in samples])
        mean_vr = np.mean([s[1] for s in samples])

        filtered_samples = [s for s in samples if
                            abs(s[0] - mean_hr) < 0.05 and abs(s[1] - mean_vr) < 0.05]

        for s in filtered_samples:
            calibration_data.append((s[0], s[1], point[0], point[1]))

    # Separar datos
    X = np.array([[d[0], d[1]] for d in calibration_data])
    y_x = np.array([d[2] for d in calibration_data])
    y_y = np.array([d[3] for d in calibration_data])

    # Entrenar Random Forest
    rf_x = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2)
    rf_y = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2)
    rf_x.fit(X, y_x)
    rf_y.fit(X, y_y)

    # Predecir sobre los datos de calibración
    pred_x = rf_x.predict(X)
    pred_y = rf_y.predict(X)

    # Calcular métricas de error
    rmse_x = math.sqrt(mean_squared_error(y_x, pred_x))
    rmse_y = math.sqrt(mean_squared_error(y_y, pred_y))

    errors = np.sqrt((y_x - pred_x) ** 2 + (y_y - pred_y) ** 2)
    mean_error = np.mean(errors)
    precision = 100 - mean_error

    print(f"\nPrecisión de calibración:")
    print(f"RMSE en X: {rmse_x:.2f} píxeles")
    print(f"RMSE en Y: {rmse_y:.2f} píxeles")
    # print(f"Error medio euclidiano (distancia): {mean_error:.2f} píxeles")

    # Mostrar gráfico de calibración (opcional)
    real_points = []
    seen_points = []

    start = 0
    print("\nResumen de calibración (punto real vs punto visto):")
    print("Idx\tReal (x,y)\t\tVisto (x,y)")

    for idx, point in enumerate(calibration_points):
        end = start + len([d for d in calibration_data[start:] if (d[2], d[3]) == point])
        samples = calibration_data[start:end]
        start = end

        if not samples:
            continue

        mean_hr = np.mean([s[0] for s in samples])
        mean_vr = np.mean([s[1] for s in samples])

        seen_x = int(rf_x.predict([[mean_hr, mean_vr]])[0])
        seen_y = int(rf_y.predict([[mean_hr, mean_vr]])[0])

        real_points.append(point)
        seen_points.append((seen_x, seen_y))

        print(f"{idx+1}\t{point}\t\t({seen_x}, {seen_y})")

    real_x, real_y = zip(*real_points)
    seen_x, seen_y = zip(*seen_points)

    resultados_path = os.path.join(output_path, f"Resultados_{codigo}")
    os.makedirs(resultados_path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(real_x, real_y, c='green', label='Puntos de referencia', s=100)
    plt.scatter(seen_x, seen_y, c='red', label='Puntos Vistos', s=100)

    for rx, ry, sx, sy in zip(real_x, real_y, seen_x, seen_y):
        plt.plot([rx, sx], [ry, sy], 'gray', linestyle='--', linewidth=1)

    plt.title("Resultados de la calibración")
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.tight_layout()
    plt.show(block=False)  # Mostrar sin bloquear el código
    plt.pause(3)           # Esperar 3 segundos antes de cerrar la gráfica
    ruta = os.path.join(resultados_path,f"Calibracion_{codigo}.png")
    plt.savefig(ruta)
    plt.close() 

    def mostrar_mensaje_calibracion(mean_error):
        # Crear ventana raíz
        root = tk.Tk()
        root.title("Resultado de Calibración")
        root.attributes("-fullscreen", True)
        root.configure(bg="#f4f4f4")  # Fondo blanco suave
        root.attributes("-topmost", True)  # Siempre al frente
        

        if precision >= 80.0:  #Este valor es en pixeles
            mensaje = "✅ Calibración exitosa.\nPuedes continuar."
            color_texto = "#2e7d32"  # Verde oscuro
        else:
            mensaje = "❌ Calibración fallida.\nRepite el proceso."
            color_texto = "#c62828"  # Rojo oscuro

        # Etiqueta de mensaje
        label = tk.Label(
            root,
            text=mensaje,
            font=("Century Gothic", 40, "bold"),
            fg=color_texto,
            bg="#f4f4f4",
            wraplength=root.winfo_screenwidth() - 100,
            justify="center"
        )
        label.pack(expand=True)

        # Cierra automáticamente después de 3 segundos
        root.after(2000, root.destroy)
        root.mainloop()

    mostrar_mensaje_calibracion(mean_error)
    #cap.release()
    cv2.destroyAllWindows()

    return rf_x, rf_y, precision, mean_error