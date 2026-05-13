import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api.dependencies import get_product_pipeline
from app.core.config import Settings
from app.main import create_app
from app.services.detector import RegionDetection
from app.services.normalizer import ProductTextNormalizer
from app.services.ocr import OcrResult, OcrTextLine
from app.services.pipeline import ProductRecognitionPipeline


class FakeDetector:
    def detect(self, image: np.ndarray) -> RegionDetection:
        height, width = image.shape[:2]
        return RegionDetection(
            bbox=(0, 0, width, height),
            model="fake-yolo-test.pt",
            confidence=0.91,
            class_id=0,
            class_name="product_label",
        )


class FakeOcr:
    def extract(self, image: np.ndarray) -> OcrResult:
        lines = [
            OcrTextLine("Inca Kola", 0.97),
            OcrTextLine("Botella 500 ml", 0.93),
        ]
        return OcrResult(
            engine="fake-ocr",
            text="\n".join(line.text for line in lines),
            average_confidence=0.95,
            lines=lines,
        )


def _jpeg_bytes() -> bytes:
    image = np.full((240, 360, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 35), (325, 205), (20, 120, 20), -1)
    cv2.putText(image, "Inca Kola", (58, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3)
    cv2.putText(image, "Botella 500 ml", (58, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _blurry_jpeg_bytes() -> bytes:
    image = cv2.imdecode(np.frombuffer(_jpeg_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    small = cv2.resize(image, (12, 8), interpolation=cv2.INTER_AREA)
    blurred = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(blurred, (121, 121), 0)
    ok, encoded = cv2.imencode(".jpg", blurred)
    assert ok
    return encoded.tobytes()


def test_recognize_product_returns_structured_json() -> None:
    app = create_app()
    fake_pipeline = ProductRecognitionPipeline(
        settings=Settings(),
        detector=FakeDetector(),
        ocr=FakeOcr(),
        normalizer=ProductTextNormalizer(),
    )
    app.dependency_overrides[get_product_pipeline] = lambda: fake_pipeline
    client = TestClient(app)

    response = client.post(
        "/api/v1/products/recognize",
        files={"image": ("inca-kola.jpg", _jpeg_bytes(), "image/jpeg")},
        headers={"X-Trace-ID": "test-trace-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"] == "test-trace-1"
    assert body["producto"]["marca"] == "Inca Kola"
    assert body["producto"]["tipo_producto"] == "Gaseosa"
    assert body["producto"]["contenido_neto"] == "500 ml"
    assert body["producto"]["categoria_sugerida"] == "bebidas"
    assert body["deteccion"]["model"] == "fake-yolo-test.pt"
    assert body["ocr"]["engine"] == "fake-ocr"


def test_root_serves_product_upload_screen() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Registro inteligente de productos" in response.text
    assert "static/app.js" in response.text
    assert "product-name-input" in response.text


def test_productos_screen_is_served() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/productos")

    assert response.status_code == 200
    assert "Editar producto" in response.text
    assert "static/productos.js" in response.text


def test_admin_screen_is_served() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/admin")

    assert response.status_code == 200
    assert "VisionAI Admin" in response.text
    assert "static/admin.js" in response.text


def test_recognize_product_creates_pending_review_event() -> None:
    app = create_app()
    fake_pipeline = ProductRecognitionPipeline(
        settings=Settings(),
        detector=FakeDetector(),
        ocr=FakeOcr(),
        normalizer=ProductTextNormalizer(),
    )
    app.dependency_overrides[get_product_pipeline] = lambda: fake_pipeline
    client = TestClient(app)

    response = client.post(
        "/api/v1/products/recognize",
        files={"image": ("inca-kola.jpg", _jpeg_bytes(), "image/jpeg")},
        headers={"X-Trace-ID": "test-review-1"},
    )
    assert response.status_code == 200

    events = client.get("/api/v1/admin/reconocimientos").json()["items"]
    assert len(events) == 1
    assert events[0]["trace_id"] == "test-review-1"
    assert events[0]["status"] == "pending_review"
    assert events[0]["predicted_marca"] == "Inca Kola"


def test_recognize_product_rejects_non_image_upload() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/v1/products/recognize",
        files={"image": ("data.txt", b"hola", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_IMAGE"


def test_recognize_product_rejects_blurry_image_before_ocr() -> None:
    app = create_app()
    fake_pipeline = ProductRecognitionPipeline(
        settings=Settings(),
        detector=FakeDetector(),
        ocr=FakeOcr(),
        normalizer=ProductTextNormalizer(),
    )
    app.dependency_overrides[get_product_pipeline] = lambda: fake_pipeline
    client = TestClient(app)

    response = client.post(
        "/api/v1/products/recognize",
        files={"image": ("borrosa.jpg", _blurry_jpeg_bytes(), "image/jpeg")},
        headers={"X-Trace-ID": "test-blurry-1"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "BLURRY_IMAGE"
    assert "borrosa" in body["message"].lower()
