# Actividad 19: Detección de Objetos en Tiempo Real con YOLOv8 y OpenCV

**Tecnológico Nacional de México**  
**Instituto Tecnológico Superior de Uruapan**  
**División de Estudios de Posgrado e Investigación**  
**Maestría en Inteligencia Artificial**  

* **Asignatura:** Inteligencia Artificial y su Ética  
* **Alumno:** Juan Pablo Figueroa Moran  
* **Matrícula:** M26040059  

---

## 📌 1. Descripción del Proyecto

Este proyecto implementa un sistema de **visión artificial en tiempo real** para la detección y localización multiclase de objetos mediante el modelo de estado del arte **YOLOv8** (`ultralytics`) integrado con `OpenCV`.

### Características Principales:
* Procesamiento de flujo de video en vivo (webcam nativa `0`, cámara externa o archivo de video local).
* Delineado dinámico de recuadros delimitadores (*bounding boxes*), etiquetas de clase y porcentaje de confianza probabilística.
* Medición y renderizado del rendimiento computacional en fotogramas por segundo (**FPS**).
* Opción de codificación y guardado del video procesado en formato MP4 (`output_detection.mp4`).
* Exportación estructurada de estadísticas acumuladas y conteos por categoría en formato JSON (`reporte_detecciones.json`).
* Manejador de respaldo sintético (*graceful fallback*) para entornos de servidor o estaciones de trabajo sin webcam física conectada.

---

## 🚀 2. Instalación y Ejecución

### Instalación de dependencias
```bash
pip install -r requirements.txt
```

### Ejecutar con cámara web por defecto (0)
```bash
python yolo_detector.py
```

### Ejecutar con un video específico
```bash
python yolo_detector.py mi_video.mp4
```

---

## 📊 3. Ejemplo de Reporte Generado (`reporte_detecciones.json`)

```json
{
  "fecha": "2026-10-05T10:15:30",
  "total_frames_procesados": 150,
  "conteo_total_por_clase": {
    "person": 142,
    "cell phone": 87,
    "laptop": 56
  },
  "tasa_promedio_fps": 28.4
}
```

---

## ⚖️ 4. Consideraciones Éticas en Videovigilancia y Visión Computacional

1. **Derecho a la Privacidad:** La captura continua de imágenes en espacios públicos o laborales puede vulnerar la intimidad individual. Debe regirse por el principio de proporcionalidad y aviso previo de privacidad.
2. **Sesgo en Detectores:** Los modelos convolucionales pueden presentar tasas variables de acierto según tono de piel, vestimenta o iluminación, lo que exige auditorías continuas de equidad en su despliegue operativo.
