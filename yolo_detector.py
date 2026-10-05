"""
=============================================================================
TECNOLÓGICO NACIONAL DE MÉXICO - INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN
DIVISIÓN DE ESTUDIOS DE POSGRADO E INVESTIGACIÓN
MAESTRÍA EN INTELIGENCIA ARTIFICIAL

Materia: Inteligencia Artificial y su Ética
Actividad 19: Visión Artificial - Detección de Objetos con YOLOv8 en Tiempo Real
Alumno: Juan Pablo Figueroa Moran (Matrícula: M26040059)
=============================================================================
"""

import sys
import time
import json
import csv
import cv2
import numpy as np
from datetime import datetime
from collections import Counter

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def generar_frame_sintetico(frame_idx: int) -> np.ndarray:
    """Genera un fotograma sintético cuando no se dispone de cámara web física."""
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    # Dibujar objetos sintéticos en movimiento
    pos_x = int(100 + 150 * np.sin(frame_idx * 0.1))
    cv2.circle(frame, (pos_x, 240), 60, (0, 200, 255), -1) # 'pelota / objeto'
    cv2.putText(frame, "MODO SIMULACION - WEBCAM NO DISPONIBLE", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_idx}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame


def ejecutar_detector_yolo(source=0,
                           model_name="yolov8n.pt",
                           guardar_video=True,
                           output_video_path="output_detection.mp4",
                           reporte_path="reporte_detecciones.json",
                           max_frames=150):
    """
    Ejecuta detección de objetos en tiempo real con YOLOv8:
    - Bounding boxes con etiquetas de clase y confianza (%)
    - Contador de FPS (fotogramas por segundo)
    - Grabación opcional de video resultante
    - Exportación de reporte consolidado con conteo de clases detectadas
    """
    print("=" * 75)
    print("  TECNM / ITSU - DETECTOR DE OBJETOS EN TIEMPO REAL (YOLOv8)")
    print("=" * 75)

    if YOLO is None:
        print("[!] La librería 'ultralytics' no está instalada. Ejecute: pip install ultralytics")
        return

    print(f"[*] Cargando modelo YOLOv8 ({model_name})...")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(source)
    usar_sintetico = False

    if not cap.isOpened():
        print(f"[!] No fue posible abrir la fuente de video '{source}'. Activando generador de prueba sintético...")
        usar_sintetico = True

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if not usar_sintetico else 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if not usar_sintetico else 480
    fps_in = cap.get(cv2.CAP_PROP_FPS) if not usar_sintetico else 25.0
    if fps_in <= 0:
        fps_in = 25.0

    video_writer = None
    if guardar_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps_in, (width, height))
        print(f"[+] Video procesado será guardado en: '{output_video_path}'")

    registro_detecciones = []
    conteo_global_clases = Counter()
    
    frame_idx = 0
    t_prev = time.time()

    print("[*] Iniciando procesamiento de video... (Presione 'q' para detener)")

    try:
        while frame_idx < max_frames:
            if not usar_sintetico:
                ret, frame = cap.read()
                if not ret:
                    print("[*] Fin del flujo de video.")
                    break
            else:
                frame = generar_frame_sintetico(frame_idx)
                time.sleep(0.03)

            # Inferencia YOLOv8
            results = model(frame, verbose=False)[0]

            # Conteo de FPS
            t_curr = time.time()
            fps_actual = 1.0 / (t_curr - t_prev) if (t_curr - t_prev) > 0 else 0
            t_prev = t_curr

            detecciones_en_frame = []

            # Dibujar bounding boxes y etiquetas
            for box in results.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = model.names[cls_id]

                conteo_global_clases[cls_name] += 1
                detecciones_en_frame.append({
                    "clase": cls_name,
                    "confianza": round(conf, 4),
                    "box": coords.tolist()
                })

                # Bounding box
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                etiqueta = f"{cls_name} {conf*100:.1f}%"
                cv2.putText(frame, etiqueta, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Superponer FPS en pantalla
            cv2.putText(frame, f"FPS: {fps_actual:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Guardar frame si aplica
            if video_writer:
                video_writer.write(frame)

            registro_detecciones.append({
                "frame": frame_idx,
                "timestamp": datetime.now().isoformat(),
                "detecciones": detecciones_en_frame
            })

            # Mostrar ventana solo si hay entorno gráfico interactivo
            try:
                cv2.imshow("YOLOv8 Real-Time Detection - TecNM / ITSU", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] Proceso interrumpido por el usuario.")
                    break
            except Exception:
                # Entornos headless sin GUI
                pass

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"    -> Procesados {frame_idx}/{max_frames} fotogramas. FPS: {fps_actual:.1f}")

    finally:
        if not usar_sintetico:
            cap.release()
        if video_writer:
            video_writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Exportar reporte de detecciones acumuladas
    reporte_final = {
        "fecha": datetime.now().isoformat(),
        "total_frames_procesados": frame_idx,
        "conteo_total_por_clase": dict(conteo_global_clases),
        "tasa_promedio_fps": round(fps_in, 2),
        "historial_frames": registro_detecciones[:50] # muestra acotada
    }

    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(reporte_final, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Reporte acumulado exportado a: '{reporte_path}'")

    print("\n--- RESUMEN DE DETECCIONES GLOBALES ---")
    for cls, count in conteo_global_clases.items():
        print(f"  * Clase '{cls}': {count} detecciones registradas")

    print("\n" + "=" * 75)
    print("  CONSIDERACIONES ÉTICAS EN SISTEMAS DE DETECCIÓN Y VIDEOVIGILANCIA")
    print("=" * 75)
    print("""
    1. Privacidad y Consentimiento: El despliegue de sistemas de visión en tiempo real
       debe informar de manera ostensible a los transeúntes sobre la recolección de video.
    2. No Reconocimiento Facial Indiscriminado: La detección genérica de objetos no debe
       derivar en identificación biométrica masiva sin fundamento legal explícito.
    3. Minimización de Retención de Datos: Las grabaciones deben anonimizarse o eliminarse
       tras cumplir su propósito legítimo para prevenir vigilancia omnipresente.
    """)


if __name__ == "__main__":
    src = 0
    if len(sys.argv) > 1:
        # Permitir pasar archivo de video o número de cámara
        src = sys.argv[1]
        if src.isdigit():
            src = int(src)
    ejecutar_detector_yolo(source=src)
