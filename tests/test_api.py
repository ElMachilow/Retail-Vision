import sqlite3
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api.dependencies import get_product_pipeline
from app.core.config import Settings, get_settings
from app.main import create_app
from app.services.detector import FieldDetection, RegionDetection
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


class FakeFieldDetector:
    def detect(self, image: np.ndarray) -> RegionDetection:
        height, width = image.shape[:2]
        return RegionDetection(
            bbox=(0, 0, width, height),
            model="fake-fields-yolo-test.pt",
            confidence=0.93,
            class_id=None,
            class_name="brand+product_name+net_weight",
            fields=(
                FieldDetection((10, 10, 120, 60), 0.91, 0, "brand"),
                FieldDetection((10, 70, 220, 130), 0.94, 1, "product_name"),
                FieldDetection((10, 140, 180, 190), 0.90, 2, "net_weight"),
            ),
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


class FakeFieldAwareOcr:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, image: np.ndarray) -> OcrResult:
        self.calls += 1
        line_by_call = {
            1: OcrTextLine("Inca Kola", 0.97, bbox_area=8_000),
            2: OcrTextLine("Botella", 0.93, bbox_area=6_000),
            3: OcrTextLine("500 ml", 0.95, bbox_area=4_000),
            4: OcrTextLine("Inca Kola", 0.90, bbox_area=8_000),
        }
        line = line_by_call[self.calls]
        return OcrResult(engine="fake-ocr", text=line.text, average_confidence=line.confidence, lines=[line])


class FakeProminentOcr:
    def extract(self, image: np.ndarray) -> OcrResult:
        lines = [
            OcrTextLine("ANTIACIDO", 0.91, bbox_area=4_000),
            OcrTextLine("GASEOVET MS", 0.94, bbox_area=24_000),
            OcrTextLine("Magaldrato + Simeticona", 0.90, bbox_area=8_000),
            OcrTextLine("(800 mg + 60 mg)/10 mL", 0.88, bbox_area=7_000),
            OcrTextLine("Suspensión Oral", 0.89, bbox_area=5_000),
            OcrTextLine("220 mL", 0.86, bbox_area=3_000),
        ]
        return OcrResult(
            engine="fake-ocr",
            text="\n".join(line.text for line in lines),
            average_confidence=0.90,
            lines=lines,
        )


class FakeCartavioProminentOcr:
    def extract(self, image: np.ndarray) -> OcrResult:
        lines = [
            OcrTextLine("RON", 0.94, bbox_area=6_500),
            OcrTextLine("CARTAVIO", 0.96, bbox_area=24_000),
            OcrTextLine("INTENSAMENTE TOSTADO", 0.89, bbox_area=3_000),
            OcrTextLine("BLACK BARREL", 0.95, bbox_area=18_000),
            OcrTextLine("3 ANOS", 0.92, bbox_area=7_000),
            OcrTextLine("1L | 40% vol", 0.88, bbox_area=2_000),
        ]
        return OcrResult(engine="fake-ocr", text="\n".join(line.text for line in lines), average_confidence=0.92, lines=lines)


class FakeGingisonaProminentOcr:
    def extract(self, image: np.ndarray) -> OcrResult:
        lines = [
            OcrTextLine("Gingisona", 0.96, bbox_area=22_000),
            OcrTextLine("Boca y Garganta", 0.91, bbox_area=9_000),
            OcrTextLine("Antiinflamatorio Analgesico", 0.90, bbox_area=14_000),
            OcrTextLine("Bencidamina Clorhidrato 0.30%", 0.88, bbox_area=5_000),
            OcrTextLine("Solucion para pulverizacion bucal", 0.86, bbox_area=4_000),
            OcrTextLine("15 mL", 0.89, bbox_area=2_000),
        ]
        return OcrResult(engine="fake-ocr", text="\n".join(line.text for line in lines), average_confidence=0.90, lines=lines)


class FakeAvamysProminentOcr:
    def extract(self, image: np.ndarray) -> OcrResult:
        lines = [
            OcrTextLine("0075143", 0.90, bbox_area=5_000),
            OcrTextLine("Avamys", 0.96, bbox_area=18_000),
            OcrTextLine("Furoato de Fluticasona", 0.92, bbox_area=12_000),
            OcrTextLine("Spray nasal / Nasal spray", 0.90, bbox_area=7_000),
            OcrTextLine("27,5 mcg / dosis / dose", 0.88, bbox_area=5_000),
            OcrTextLine("120", 0.86, bbox_area=6_000),
        ]
        return OcrResult(engine="fake-ocr", text="\n".join(line.text for line in lines), average_confidence=0.90, lines=lines)


def _jpeg_bytes() -> bytes:
    image = np.full((240, 360, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 35), (325, 205), (20, 120, 20), -1)
    cv2.putText(image, "Inca Kola", (58, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3)
    cv2.putText(image, "Botella 500 ml", (58, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _login_admin(client: TestClient) -> None:
    response = client.post(
        "/login",
        data={"username": "admin", "password": "admin123", "next": "/admin"},
        follow_redirects=False,
    )
    assert response.status_code == 303


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


def test_inventory_screen_is_served() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/inventario")

    assert response.status_code == 200
    assert "Inventario por foto" in response.text
    assert "static/inventario.js" in response.text


def test_admin_screen_requires_login() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/admin", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_docs_require_login() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/docs", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    assert client.get("/openapi.json", follow_redirects=False).status_code == 303


def test_admin_screen_is_served_after_login() -> None:
    app = create_app()
    client = TestClient(app)
    _login_admin(client)

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

    _login_admin(client)
    events = client.get("/api/v1/admin/reconocimientos").json()["items"]
    assert len(events) == 1
    assert events[0]["trace_id"] == "test-review-1"
    assert events[0]["status"] == "pending_review"
    assert events[0]["predicted_marca"] == "Inca Kola"

    settings = get_settings()
    with sqlite3.connect(settings.sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT image_blob, image_path FROM recognition_events").fetchone()
    assert row["image_blob"] is None
    assert row["image_path"]

    image_response = client.get(f"/api/v1/admin/reconocimientos/{events[0]['id']}/image")
    assert image_response.status_code == 200
    assert image_response.content.startswith(b"\xff\xd8")


def test_inventory_recognize_photo_returns_inventory_payload() -> None:
    app = create_app()
    fake_pipeline = ProductRecognitionPipeline(
        settings=Settings(),
        detector=FakeDetector(),
        ocr=FakeOcr(),
        normalizer=ProductTextNormalizer(),
    )
    app.dependency_overrides[get_product_pipeline] = lambda: fake_pipeline
    client = TestClient(app)
    session = client.post(
        "/api/v1/inventory/sessions",
        json={"nombre": "Inventario con foto"},
    ).json()

    response = client.post(
        f"/api/v1/inventory/sessions/{session['id']}/items/recognize",
        files={"image": ("inca-kola.jpg", _jpeg_bytes(), "image/jpeg")},
        headers={"X-Trace-ID": "inventory-test-1"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["trace_id"] == "inventory-test-1"
    assert body["recognition_event_id"]
    assert body["producto"]["categoria_sugerida"] == "bebidas"
    assert body["image_url"].endswith(f"/{body['recognition_event_id']}/image")


def test_product_stock_count_confirmation_persists_photo_evidence() -> None:
    app = create_app()
    fake_pipeline = ProductRecognitionPipeline(
        settings=Settings(),
        detector=FakeDetector(),
        ocr=FakeOcr(),
        normalizer=ProductTextNormalizer(),
    )
    app.dependency_overrides[get_product_pipeline] = lambda: fake_pipeline
    client = TestClient(app)

    recognition_ids = []
    for index in range(2):
        response = client.post(
            "/api/v1/products/recognize",
            files={"image": (f"inca-kola-{index}.jpg", _jpeg_bytes(), "image/jpeg")},
            headers={"X-Trace-ID": f"stock-count-{index}"},
        )
        assert response.status_code == 200, response.text
        recognition_ids.append(response.json()["recognition_event_id"])

    response = client.post(
        "/api/v1/inventory/product-stock-counts",
        json={
            "mobile_product_id": "42",
            "nombre_producto": "Inca Kola 500 ml",
            "cantidad_final": 2,
            "confianza": 0.93,
            "photos": [
                {
                    "recognition_event_id": recognition_ids[0],
                    "source_name": "inca-kola-0.jpg",
                    "detected_name": "Inca Kola 500 ml",
                    "matched": True,
                    "accepted": True,
                    "confidence": 0.95,
                    "warnings": [],
                },
                {
                    "recognition_event_id": recognition_ids[1],
                    "source_name": "inca-kola-1.jpg",
                    "detected_name": "Inca Kola 500 ml",
                    "matched": True,
                    "accepted": True,
                    "confidence": 0.91,
                    "warnings": [],
                },
            ],
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["mobile_product_id"] == "42"
    assert body["cantidad_final"] == 2
    assert body["total_fotos"] == 2
    assert body["valid_fotos"] == 2
    assert [photo["recognition_event_id"] for photo in body["photos"]] == recognition_ids


def test_pipeline_uses_field_aware_ocr_when_yolo_returns_fields() -> None:
    ocr = FakeFieldAwareOcr()
    pipeline = ProductRecognitionPipeline(
        settings=Settings(),
        detector=FakeFieldDetector(),
        ocr=ocr,
        normalizer=ProductTextNormalizer(),
    )

    result = pipeline.process(
        image_bytes=_jpeg_bytes(),
        trace_id="test-fields-1",
        source_name="inca-kola.jpg",
    )

    assert ocr.calls == 4
    assert result.ocr.text.splitlines() == ["Inca Kola", "Botella", "500 ml"]
    assert result.producto.contenido_neto == "500 ml"


def test_recognize_product_uses_largest_ocr_text_as_required_name_candidate() -> None:
    app = create_app()
    fake_pipeline = ProductRecognitionPipeline(
        settings=Settings(),
        detector=FakeDetector(),
        ocr=FakeProminentOcr(),
        normalizer=ProductTextNormalizer(),
    )
    app.dependency_overrides[get_product_pipeline] = lambda: fake_pipeline
    client = TestClient(app)

    response = client.post(
        "/api/v1/products/recognize",
        files={"image": ("gaseovet.jpg", _jpeg_bytes(), "image/jpeg")},
        headers={"X-Trace-ID": "test-prominent-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ocr"]["prominent_text"] == "GASEOVET MS"
    assert body["producto"]["nombre_producto"] == "Gaseovet Ms 220 ml"


def test_recognize_product_builds_name_from_multiple_large_ocr_lines() -> None:
    app = create_app()
    fake_pipeline = ProductRecognitionPipeline(
        settings=Settings(),
        detector=FakeDetector(),
        ocr=FakeCartavioProminentOcr(),
        normalizer=ProductTextNormalizer(),
    )
    app.dependency_overrides[get_product_pipeline] = lambda: fake_pipeline
    client = TestClient(app)

    response = client.post(
        "/api/v1/products/recognize",
        files={"image": ("cartavio.jpg", _jpeg_bytes(), "image/jpeg")},
        headers={"X-Trace-ID": "test-cartavio-prominent"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ocr"]["prominent_text"] == "RON CARTAVIO BLACK BARREL 3 ANOS"
    assert body["producto"]["nombre_producto"].startswith("Ron Cartavio Black Barrel")
    assert body["producto"]["categoria_sugerida"] == "bebidas"


def test_recognize_product_prefers_prominent_pharmacy_brand_name() -> None:
    app = create_app()
    fake_pipeline = ProductRecognitionPipeline(
        settings=Settings(),
        detector=FakeDetector(),
        ocr=FakeGingisonaProminentOcr(),
        normalizer=ProductTextNormalizer(),
    )
    app.dependency_overrides[get_product_pipeline] = lambda: fake_pipeline
    client = TestClient(app)

    response = client.post(
        "/api/v1/products/recognize",
        files={"image": ("gingisona.jpg", _jpeg_bytes(), "image/jpeg")},
        headers={"X-Trace-ID": "test-gingisona-prominent"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ocr"]["prominent_text"].startswith("Gingisona")
    assert body["producto"]["nombre_producto"].startswith("Gingisona")
    assert body["producto"]["categoria_sugerida"] == "farmacia/otc"


def test_recognize_product_keeps_brand_only_prominent_name_for_pharmacy() -> None:
    app = create_app()
    fake_pipeline = ProductRecognitionPipeline(
        settings=Settings(),
        detector=FakeDetector(),
        ocr=FakeAvamysProminentOcr(),
        normalizer=ProductTextNormalizer(),
    )
    app.dependency_overrides[get_product_pipeline] = lambda: fake_pipeline
    client = TestClient(app)

    response = client.post(
        "/api/v1/products/recognize",
        files={"image": ("avamys.jpg", _jpeg_bytes(), "image/jpeg")},
        headers={"X-Trace-ID": "test-avamys-prominent"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ocr"]["prominent_text"].startswith("Avamys")
    assert body["producto"]["nombre_producto"].startswith("Avamys")
    assert body["producto"]["categoria_sugerida"] == "farmacia/otc"


def test_delete_recognition_removes_admin_event() -> None:
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
        headers={"X-Trace-ID": "test-delete-1"},
    )
    assert response.status_code == 200
    _login_admin(client)
    event = client.get("/api/v1/admin/reconocimientos").json()["items"][0]
    settings = get_settings()
    with sqlite3.connect(settings.sqlite_path) as conn:
        image_path = Path(conn.execute("SELECT image_path FROM recognition_events").fetchone()[0])
    assert image_path.exists()

    delete_response = client.delete(f"/api/v1/admin/reconocimientos/{event['id']}")

    assert delete_response.status_code == 204
    assert not image_path.exists()
    assert client.get("/api/v1/admin/reconocimientos").json()["items"] == []
    assert client.get(f"/api/v1/admin/reconocimientos/{event['id']}").status_code == 404


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
