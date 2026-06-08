from fastapi.testclient import TestClient

from app.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _create_payload(**overrides) -> dict:
    payload = {
        "nombre_producto": "Inca Kola Botella 500 ml",
        "marca": "Inca Kola",
        "tipo_producto": "Gaseosa",
        "presentacion": "botella 500 ml",
        "contenido_neto": "500 ml",
        "unidad_medida": "ml",
        "categoria_sugerida": "bebidas",
        "codigo_barras": "7751234500011",
        "precio_venta": 4.5,
    }
    payload.update(overrides)
    return payload


def test_create_and_get_product() -> None:
    client = _client()
    response = client.post("/api/v1/productos", json=_create_payload())
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"]
    assert body["precio_venta"] == 4.5

    fetched = client.get(f"/api/v1/productos/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["nombre_producto"] == "Inca Kola Botella 500 ml"


def test_create_rejects_missing_required_fields() -> None:
    client = _client()
    response = client.post(
        "/api/v1/productos",
        json={"nombre_producto": "  ", "categoria_sugerida": "bebidas", "precio_venta": 1.0},
    )
    assert response.status_code == 422


def test_categorize_product_from_name() -> None:
    client = _client()
    response = client.post(
        "/api/v1/productos/categorize",
        json={"nombre_producto": "Detergente Ariel 1 kg"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["marca"] == "Ariel"
    assert body["tipo_producto"] == "Detergente"
    assert body["categoria_sugerida"] == "limpieza"


def test_categorize_product_uses_ocr_context() -> None:
    client = _client()
    response = client.post(
        "/api/v1/productos/categorize",
        json={
            "nombre_producto": "Aftatopic 10 ml",
            "context": "AFTATOPIC SOLUCION Composicion Acidosalicilico Excipientes c.s.p. 10 ml",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["categoria_sugerida"] == "farmacia/otc"
    assert body["tipo_producto"] == "Solucion"


def test_create_infers_category_when_missing() -> None:
    client = _client()
    payload = _create_payload(
        nombre_producto="Detergente Ariel 1 kg",
        marca=None,
        tipo_producto=None,
        categoria_sugerida=None,
        codigo_barras="7751234500099",
    )
    response = client.post("/api/v1/productos", json=payload)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["marca"] == "Ariel"
    assert body["tipo_producto"] == "Detergente"
    assert body["categoria_sugerida"] == "limpieza"


def test_create_rejects_negative_price() -> None:
    client = _client()
    response = client.post("/api/v1/productos", json=_create_payload(precio_venta=-1))
    assert response.status_code == 422


def test_duplicate_barcode_returns_409() -> None:
    client = _client()
    first = client.post("/api/v1/productos", json=_create_payload())
    assert first.status_code == 201
    second = client.post(
        "/api/v1/productos",
        json=_create_payload(nombre_producto="Otro nombre"),
    )
    assert second.status_code == 409
    assert second.json()["error_code"] == "DUPLICATE_BARCODE"


def test_update_product_modifies_fields() -> None:
    client = _client()
    created = client.post("/api/v1/productos", json=_create_payload()).json()
    response = client.put(
        f"/api/v1/productos/{created['id']}",
        json=_create_payload(precio_venta=5.9, nombre_producto="Inca Kola 500 ml"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["precio_venta"] == 5.9
    assert body["nombre_producto"] == "Inca Kola 500 ml"


def test_update_missing_product_returns_404() -> None:
    client = _client()
    response = client.put("/api/v1/productos/9999", json=_create_payload())
    assert response.status_code == 404
    assert response.json()["error_code"] == "PRODUCT_NOT_FOUND"


def test_suggestions_below_three_chars_returns_empty() -> None:
    client = _client()
    response = client.get("/api/v1/productos/suggestions", params={"q": "in"})
    assert response.status_code == 200
    assert response.json()["items"] == []


def test_suggestions_returns_db_match_first() -> None:
    client = _client()
    client.post("/api/v1/productos", json=_create_payload())
    response = client.get("/api/v1/productos/suggestions", params={"q": "inca"})
    assert response.status_code == 200
    items = response.json()["items"]
    assert items, items
    assert items[0]["source"] == "db"
    assert "Inca Kola" in items[0]["nombre_producto"]


def test_suggestions_falls_back_to_catalog() -> None:
    client = _client()
    response = client.get("/api/v1/productos/suggestions", params={"q": "gas"})
    assert response.status_code == 200
    items = response.json()["items"]
    assert items, items
    assert all(item["source"] == "catalog" for item in items)
    assert any(item["nombre_producto"].lower().startswith("gaseosa") for item in items)


def test_suggestions_prioritize_ocr_context() -> None:
    client = _client()
    context = "boreal\nDetergente\nliquido\nAroma\nLawanda\nCo3L"

    response = client.get(
        "/api/v1/productos/suggestions",
        params={"q": "Detergente", "context": context, "source_name": "boreal.png"},
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert items, items
    assert items[0]["source"] == "ocr"
    assert items[0]["nombre_producto"] == "Detergente Boreal Lavanda"
    assert items[0]["marca"] == "Boreal"


def test_suggestions_returns_three_ocr_options_from_context() -> None:
    client = _client()
    context = (
        "ABRIRAOU\nNUEVA\nIMAGEN\nDETERGENTE\nROSAS Y MAGNOLIAS\n"
        "Bolivar\nCuidadoy\nSuavidad\nSUAVIDAD Y\ncontogue\n"
        "FRAGANCIA\nde suav zante\nycapsulas\nPROLONGADA\ndearoma"
    )

    response = client.get(
        "/api/v1/productos/suggestions",
        params={
            "q": "Detergente Bolívar Morado",
            "context": context,
            "source_name": "bol-morado_png.rf.8deff10bb587d9bb8525ae403509c340.jpg",
        },
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 3
    assert all(item["source"] == "ocr" for item in items)
    assert items[0]["nombre_producto"] == "Detergente Bolívar Rosas y Magnolias"
    assert items[1]["nombre_producto"] == "Detergente Bolívar Cuidado y Suavidad"
    assert items[2]["nombre_producto"] == "Detergente Bolívar Suavidad y Fragancia Prolongada"


def test_suggestions_creates_creative_bolivar_options_from_context() -> None:
    client = _client()
    context = (
        "NUEVA\nIMAGEN\nDCJERSENTEF!CFJE\nw 4kg\nBolivar\n"
        "Cuidado\nTotal\nPROTEGE\ncontsarculas\nELCOLORY\n"
        "protictoras\nLASFIBRAS"
    )

    response = client.get(
        "/api/v1/productos/suggestions",
        params={
            "q": "Jabón Bolívar 4 kg",
            "context": context,
            "source_name": "bolivar_png.rf.cde8895cbd7d7959fd91d92a9cc6423e.jpg",
        },
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 3
    assert items[0]["nombre_producto"] == "Jabón Bolívar Cuidado Total"
    assert items[1]["nombre_producto"] == "Jabón Bolívar Protege Color y Fibras"
    assert items[2]["nombre_producto"] == "Jabón Bolívar Cuidado Total 4 kg"


def test_suggestions_prioritize_prominent_pharmacy_name_from_context() -> None:
    client = _client()
    context = (
        "Dextrometorfano\n"
        "15mg/5mL\n"
        "Jarabe-Viaoral\n"
        "Ventas\n"
        "Mane\n"
        "Supresor de la tos\n"
        "120ml\n"
        "Proley\n"
        "Medifarma"
    )

    response = client.get(
        "/api/v1/productos/suggestions",
        params={"q": "Jarabe Viaoral Medifarma", "context": context, "source_name": "dextri.jpeg"},
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert items[0]["nombre_producto"] == "Dextrometorfano Jarabe Medifarma"


def test_suggestions_always_builds_three_options_from_ocr_text_when_possible() -> None:
    client = _client()
    context = (
        "Rinoval\n"
        "MOMETASONA FUROATO\n"
        "50 mcg/dosis\n"
        "120 dosis\n"
        "Suspensión Nasal para Nebulización\n"
        "SAVAL"
    )

    response = client.get(
        "/api/v1/productos/suggestions",
        params={
            "q": "Rinoval",
            "limit": 3,
            "context": context,
            "prominent_text": "Rinoval",
        },
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 3
    assert items[0]["nombre_producto"] == "Rinoval"
    assert any("MOMETASONA FUROATO" in item["nombre_producto"] for item in items[1:])
    assert any("Suspensión Nasal" in item["nombre_producto"] for item in items[1:])


def test_suggestions_infers_prominent_name_when_client_does_not_send_it() -> None:
    client = _client()
    context = (
        "220mL\n"
        "ANTIACIDO\n"
        "GASEOVET\"MS\n"
        "Magaldrato+Simeticona\n"
        "800mg+60 mg/10 mL\n"
        "Suspensión Oral\n"
        "Viaoral\n"
        "GASEOVET'MS\n"
        "800mg+60mg/10 ml"
    )

    response = client.get(
        "/api/v1/productos/suggestions",
        params={
            "q": "Gaseovet Ms",
            "limit": 3,
            "context": context,
            "source_name": "gaseovet.jpeg",
        },
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 3
    assert items[0]["nombre_producto"] == "Gaseovet Ms 220 ml"
    assert all(item["nombre_producto"].startswith("Gaseovet Ms") for item in items)


def test_suggestions_capped_to_three() -> None:
    client = _client()
    response = client.get(
        "/api/v1/productos/suggestions", params={"q": "ola", "limit": 10}
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) <= 10  # respects limit, but UI uses 3
    response_three = client.get(
        "/api/v1/productos/suggestions", params={"q": "ola", "limit": 3}
    )
    assert len(response_three.json()["items"]) <= 3


def test_list_products_returns_recent_first() -> None:
    client = _client()
    first = client.post("/api/v1/productos", json=_create_payload()).json()
    second = client.post(
        "/api/v1/productos",
        json=_create_payload(codigo_barras="7751234500022", nombre_producto="Field Galleta"),
    ).json()
    response = client.get("/api/v1/productos")
    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["items"]]
    assert ids[0] == second["id"]
    assert first["id"] in ids


def test_inventory_session_item_and_summary() -> None:
    client = _client()
    session_response = client.post(
        "/api/v1/inventory/sessions",
        json={"nombre": "Inventario piloto"},
    )
    assert session_response.status_code == 201, session_response.text
    session = session_response.json()

    item_response = client.post(
        f"/api/v1/inventory/sessions/{session['id']}/items",
        json={
            "nombre_producto": "Detergente Ariel 1 kg",
            "marca": "Ariel",
            "tipo_producto": "Detergente",
            "categoria": "limpieza",
            "contenido_neto": "1 kg",
            "unidad_medida": "kg",
            "cantidad": 6,
            "ubicacion": "Estante 1",
        },
    )
    assert item_response.status_code == 201, item_response.text
    item = item_response.json()
    assert item["cantidad"] == 6
    assert item["categoria"] == "limpieza"

    summary = client.get(f"/api/v1/inventory/sessions/{session['id']}/summary")
    assert summary.status_code == 200
    body = summary.json()
    assert body["total_productos"] == 1
    assert body["total_unidades"] == 6
    assert body["categorias"][0]["categoria"] == "limpieza"


def test_inventory_item_infers_category_when_missing() -> None:
    client = _client()
    session = client.post(
        "/api/v1/inventory/sessions",
        json={"nombre": "Inventario categorias"},
    ).json()

    response = client.post(
        f"/api/v1/inventory/sessions/{session['id']}/items",
        json={"nombre_producto": "Aftatopic 10 ml", "cantidad": 2},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["cantidad"] == 2
    assert body["contenido_neto"] == "10 ml"
