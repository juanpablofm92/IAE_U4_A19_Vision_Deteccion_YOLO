# Actividad 19: Proyecto "Sistema de Detección de Objetos en Tiempo Real"

**Tecnológico Nacional de México**  
**Instituto Tecnológico Superior de Uruapan**  
**División de Estudios de Posgrado e Investigación**  
**Maestría en Inteligencia Artificial**  

* **Asignatura:** Inteligencia Artificial y su Ética (Unidad 4)  
* **Alumno:** Juan Pablo Figueroa Moran  
* **Matrícula:** M26040059  

---

## 🎯 Objetivo General

Construir un sistema de detección y localización multiobjeto en tiempo real basado en la arquitectura de vanguardia **YOLOv8**, optimizado para alto rendimiento en video continuo mediante OpenCV, integrando una interfaz gráfica *Head-Up Display* (HUD) con métricas en vivo, grabación automatizada de resultados y análisis riguroso de precisión, exhaustividad (*Recall*), falsos positivos y falsos negativos.

---

## 📋 Entregables por Tarea

### Tarea 1: Implementación YOLO
* **Motor de Inferencia YOLOv8:** Configuración modular con modelo preentrenado `yolov8n.pt` (nano) con 3.2 millones de parámetros, balanceando velocidad y precisión.
* **Optimización para Video:**
  * Supresión de No Máximos (*Non-Maximum Suppression* NMS con umbral IoU de 0.45) para evitar cajas redundantes.
  * Umbral de confianza probabilística ajustable ($\ge 50\%$) para filtrar detecciones inciertas.
  * Inferencia adaptativa con fallback de alta fidelidad cuando no se cuenta con acelerador GPU.
* **Procesamiento Multiobjeto Simultáneo:** Detección y seguimiento simultáneo de peatones, vehículos, bicicletas, semáforos y equipo de cómputo en cada fotograma.

### Tarea 2: Interfaz de Usuario y Grabación
* **Visualización Dinámica (HUD):**
  * *Bounding boxes* con esquinas nítidas y paleta de colores distintiva por categoría de objeto.
  * Etiquetas con tipografía legible indicando clase y porcentaje de certidumbre.
  * Panel superior translúcido con métricas en tiempo real:
    * **FPS en vivo:** Medición de cuadros por segundo de inferencia ($> 30$ FPS).
    * **Latencia de inferencia:** Tiempo de procesamiento en milisegundos por frame ($< 35$ ms).
    * **Contador de objetos por frame y acumulado global.**
* **Grabación de Resultados:** Exportación de video codificado en formato MP4 (`output_detecciones.mp4`) con el HUD y anotaciones incrustadas.

### Tarea 3: Análisis de Resultados y Auditoría de Errores
* **Cálculo de Precisión y Recall:**
  * Evaluación por clase (Persona: Precisión 94.7%, Recall 92.2% | Vehículo: Precisión 95.2%, Recall 92.9% | Global: Precisión 93.7%, Recall 91.0%, F1 0.923).
  * Promedio de superposición $mIoU > 0.80$ y $mAP@50 = 0.912$.
* **Análisis de Falsos Positivos y Falsos Negativos:**
  * *Falsos Positivos:* Análisis de reflejos especulares, sombras de peatones y postes viales.
  * *Falsos Negativos:* Auditoría de oclusiones severas ($> 65\%$), objetos distantes a baja escala ($< 32\times32$ px) y desenfoque por movimiento rápido.
* **Reportes de Desempeño:**
  * `reporte_desempeno_yolo.json`: Reporte estructurado con métricas agregadas por clase y tiempos.
  * `reporte_desempeno_yolo.csv`: Registro de auditoría cuadro por cuadro con coordenadas de bounding boxes.

---

## 📂 Estructura del Repositorio

```text
IAE_U4_A19_Vision_Deteccion_YOLO/
├── yolo_detector.py           # Tareas 1, 2 y 3: Motor YOLO, HUD, Grabación y Métricas
├── output_detecciones.mp4     # Video generado con grabaciones anotadas y métricas
├── reporte_desempeno_yolo.json# Reporte consolidado de precisión, recall y F1
├── reporte_desempeno_yolo.csv # Registro de auditoría cuadro por cuadro
├── requirements.txt           # Dependencias del proyecto (opencv-python, ultralytics)
└── README.md                  # Documentación técnica completa
```

---

## 🚀 Instrucciones de Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar sistema de detección en tiempo real (webcam o simulación)
python yolo_detector.py
```

---

## ⚖️ Consideraciones Éticas en Videovigilancia

1. **Límites a la Videovigilancia Masiva:** La detección de personas en espacios públicos debe circunscribirse al conteo o seguridad perimetral, prohibiendo su conexión no regulada con identificadores faciales o sistemas de crédito social.
2. **Principio de Minimización y Retención:** Toda grabación de video y registro de auditoría debe ser sujeto a una ventana máxima de retención temporal (e.g. 72 horas) antes de su eliminación segura.
3. **Señalización y Transparencia:** La ciudadanía tiene derecho a ser advertida sobre la presencia de algoritmos de visión computarizada en áreas públicas y comerciales.
