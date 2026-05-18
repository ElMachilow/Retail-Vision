# Tesis Retail Vision MVP

MVP para la tesis **"Sistema inteligente para optimizar el proceso de gestión e inventario de productos usando redes neuronales en empresas PYMES del sector retail"**.

El sistema recibe una imagen de producto, detecta la región relevante con **YOLO**, extrae texto con **PaddleOCR**, normaliza información frecuente en productos comercializados en Perú y devuelve una propuesta JSON para el registro inicial.

## Arquitectura

```text
app/
  api/             Endpoints FastAPI y dependencias inyectables
  core/            Configuración, logging y errores esperados
  domain/          Catálogo inicial de marcas, unidades y categorías peruanas
  schemas/         Contratos Pydantic de entrada/salida
  services/        Pipeline, YOLO, OCR, preprocesamiento y normalización
tests/             Pruebas unitarias y de API con dobles controlados
runtime/debug/     Recortes de depuración opcionales
```

Flujo principal:

1. `POST /api/v1/products/recognize` recibe una imagen.
2. OpenCV decodifica y prepara la imagen.
3. YOLO detecta la región del empaque o etiqueta.
4. Se recorta la región detectada.
5. PaddleOCR extrae líneas de texto.
6. El normalizador aplica reglas para marcas, unidades y categorías frecuentes en retail peruano.
7. La API responde con producto sugerido, metadatos de detección, OCR, warnings y `trace_id`.

## Decisiones de MVP

- Detector obligatorio: `ultralytics` con `YOLO_MODEL_PATH=yolov8n.pt` por defecto.
- OCR open source: PaddleOCR (`OCR_ENGINE=paddle`). La abreviatura `es` para español está soportada por PaddleOCR en su [documentación de modelos multilenguaje](https://www.paddleocr.ai/v2.9.1/en/ppocr/blog/multi_languages.html).
- Para un MVP, YOLOv8n permite validar el flujo extremo a extremo. Para evaluación formal de la tesis se debe reemplazar por pesos fine-tuned sobre etiquetas/empaques de productos peruanos.
- Si YOLO no detecta una caja y `ALLOW_FULL_IMAGE_FALLBACK=true`, el sistema procesa la imagen completa y agrega un warning. La llamada a YOLO sigue siendo parte obligatoria del flujo.
- No se incluyen métricas de tiempo ni exactitud inventadas; quedan preparadas para medición posterior con logs, `trace_id` y campos estructurados.

## Ejecución local

Requisitos recomendados:

- Python 3.10 o 3.11
- Entorno virtual
- CPU suficiente para YOLOv8n y PaddleOCR

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Abrir:

- API: <http://localhost:8000/api/v1/health>
- Pantalla web: <http://localhost:8000/>
- Swagger: <http://localhost:8000/docs>

## Ejecución con Docker

```bash
cp .env.example .env
docker compose up --build
```

La primera ejecución puede tardar por instalación de dependencias y descarga de modelos.

## Probar el endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/products/recognize" \
  -H "X-Trace-ID: prueba-local-001" \
  -F "image=@/ruta/a/imagen-producto.jpg"
```

Respuesta esperada, con valores dependientes de la imagen:

```json
{
  "trace_id": "prueba-local-001",
  "producto": {
    "nombre_producto": "Inca Kola",
    "marca": "Inca Kola",
    "presentacion": "botella 500 ml",
    "contenido_neto": "500 ml",
    "unidad_medida": "ml",
    "categoria_sugerida": "bebidas"
  },
  "deteccion": {
    "model": "yolov8n.pt",
    "bbox": {
      "x_min": 0,
      "y_min": 0,
      "x_max": 640,
      "y_max": 480
    },
    "confidence": 0.82,
    "class_id": 0,
    "class_name": "bottle",
    "used_full_image_fallback": false
  },
  "ocr": {
    "engine": "paddleocr",
    "text": "Inca Kola\nBotella 500 ml",
    "average_confidence": 0.95,
    "lines": []
  },
  "warnings": [],
  "processing_ms": 1200
}
```

## Pruebas

```bash
pytest
```

Las pruebas usan detectores y OCR falsos para validar el contrato del API y la normalización sin descargar pesos de YOLO ni modelos OCR.

## Entrenar YOLO con la carpeta demo

La carpeta `demo/` puede convertirse a formato YOLO con una clase `product_label`. Como las imágenes no incluyen anotaciones manuales, el script genera una caja completa por imagen:

```bash
python scripts/prepare_demo_yolo_dataset.py --source demo --output data/demo_yolo
python scripts/train_yolo_demo.py --data data/demo_yolo/dataset.yaml --epochs 15 --imgsz 416 --device cpu
```

Al terminar se copia el mejor peso a:

```text
models/product_label_demo.pt
```

Para usar ese modelo en la API:

```bash
YOLO_MODEL_PATH=models/product_label_demo.pt uvicorn app.main:app --reload
```

Esta estrategia sirve como afinamiento inicial del MVP. Para la evaluación de tesis, se deben reemplazar estas cajas completas por bounding boxes anotadas de la etiqueta o región relevante del empaque.

## Variables principales

| Variable | Descripción |
| --- | --- |
| `YOLO_MODEL_PATH` | Ruta o nombre de pesos YOLO. Por defecto `yolov8n.pt`. |
| `YOLO_CONFIDENCE_THRESHOLD` | Confianza mínima de detección. |
| `YOLO_DEVICE` | `cpu`, `mps` o `cuda` según entorno. |
| `OCR_ENGINE` | Actualmente soporta `paddle`. |
| `PADDLE_OCR_LANG` | Idioma de PaddleOCR. Por defecto `es`. |
| `ALLOW_FULL_IMAGE_FALLBACK` | Procesa imagen completa si YOLO no detecta caja. |
| `PERSIST_DEBUG_IMAGES` | Guarda recortes procesados en `runtime/debug`. |
| `CORS_ALLOW_ORIGINS` | Orígenes permitidos para CORS, separados por coma. Por defecto `*` para desarrollo local. |
| `EXPOSE_INTERNAL_ERRORS` | Si es `true`, expone detalles internos en errores 500. Por defecto `false`. |
| `RECOGNITION_IMAGE_DIR` | Carpeta donde se guardan las imágenes de reconocimientos. Por defecto `runtime/recognition-images`. |

## Pendientes técnicos

- Entrenar o ajustar YOLO con un dataset de empaques/etiquetas de productos vendidos en Perú.
- Construir dataset anotado con marcas, categorías, contenido neto y unidad de medida.
- Medir tiempo promedio de registro y exactitud contra el proceso manual.
- Añadir persistencia PostgreSQL si se requiere almacenar ejecuciones, métricas o propuestas.
- Ampliar reglas para productos a granel, etiquetas parciales y farmacia/OTC.
