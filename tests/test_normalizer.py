from app.services.normalizer import ProductTextNormalizer


def test_normalizer_detects_peruvian_brand_content_and_category() -> None:
    text = "GLORIA\nLECHE EVAPORADA ENTERA\nContenido Neto 400 g"

    result = ProductTextNormalizer().normalize(text)

    assert result.marca == "Gloria"
    assert result.tipo_producto == "Leche"
    assert result.contenido_neto == "400 g"
    assert result.unidad_medida == "g"
    assert result.categoria_sugerida == "lacteos"
    assert result.nombre_producto is not None


def test_normalizer_handles_common_beverage_units() -> None:
    text = "Inca Kola\nGaseosa sabor nacional\nBotella 1.5 L"

    result = ProductTextNormalizer().normalize(text)

    assert result.marca == "Inca Kola"
    assert result.tipo_producto == "Gaseosa"
    assert result.contenido_neto == "1.5 L"
    assert result.presentacion == "botella 1.5 L"
    assert result.categoria_sugerida == "bebidas"


def test_normalizer_prioritizes_fanta_over_packaging_noise() -> None:
    text = "CAFFEINE\n100%\nFREE\nfanta\nORANGE"

    result = ProductTextNormalizer().normalize(text, source_name="0020_fanta_na.png")

    assert result.nombre_producto == "Gaseosa Fanta Naranja"
    assert result.marca == "Fanta"
    assert result.tipo_producto == "Gaseosa"
    assert result.categoria_sugerida == "bebidas"


def test_normalizer_identifies_yogurt_as_product_type_not_only_brand() -> None:
    text = "GLORIA\nYogurt fresa\n1 kg"

    result = ProductTextNormalizer().normalize(text)

    assert result.nombre_producto == "Yogurt Gloria Fresa"
    assert result.marca == "Gloria"
    assert result.tipo_producto == "Yogurt"
    assert result.categoria_sugerida == "lacteos"


def test_normalizer_identifies_mantequilla_as_product_type() -> None:
    text = "Mantequilla Gloria con sal 200 g"

    result = ProductTextNormalizer().normalize(text)

    assert result.nombre_producto == "Mantequilla Gloria"
    assert result.marca == "Gloria"
    assert result.tipo_producto == "Mantequilla"


def test_normalizer_detects_bells_sugar_from_ocr_text() -> None:
    text = "Bell's\nAZUCAR\nBLANCA\nPESO NETO 5Kg"

    result = ProductTextNormalizer().normalize(text)

    assert result.nombre_producto == "Azúcar Bell's Blanca"
    assert result.marca == "Bell's"
    assert result.tipo_producto == "Azúcar"
    assert result.contenido_neto == "5 kg"
    assert result.unidad_medida == "kg"
    assert result.categoria_sugerida == "abarrotes"
    assert "No se identificó marca" not in " ".join(result.warnings)
