import unicodedata

from app.services.normalizer import ProductTextNormalizer


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def test_normalizer_ignores_octagon_warning_as_product_type() -> None:
    text = "Nestle\nMorochas\nSNACK\nALTO EN AZUCAR\nEVITAR SU CONSUMO EXCESIVO"

    result = ProductTextNormalizer().normalize(text, source_name="morochas.png")

    assert _fold(result.tipo_producto or "") != "azucar"
    assert result.categoria_sugerida == "snacks y golosinas"


def test_normalizer_ignores_split_octagon_warning_fragments() -> None:
    text = "Morochas\nALTO EN\nGRASAS SATURADAS\nALTO EN\nAZUCAR\nSNACK"

    result = ProductTextNormalizer().normalize(text, source_name="morochas.png")

    assert _fold(result.tipo_producto or "") != "azucar"
    assert result.categoria_sugerida == "snacks y golosinas"


def test_normalizer_ignores_joined_octagon_ocr_noise() -> None:
    text = (
        "ALTOEN\nALTOEN\nGRASAS\nSATURADAS\nAZUCAR\nLVITAHSUCONSUIO\nETCISVO\n"
        "Nestla\nMorochas\nSNACK\nSiua\nRE8\nperciaes\n429\n71\n16\n"
        "Oaetassahoravanilla\nbalndscoapataabo.chacolate"
    )

    result = ProductTextNormalizer().normalize(text, source_name="morochas.png")

    assert result.nombre_producto == "Morochas Snack Nestlé"
    assert result.marca == "Nestlé"
    assert _fold(result.tipo_producto or "") == "snack"
    assert result.contenido_neto == "42 g"
    assert result.unidad_medida == "g"
    assert result.categoria_sugerida == "snacks y golosinas"


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


def test_normalizer_puts_sports_drink_brand_first() -> None:
    text = "Contenido neto\n500ml\nSPORADE\nTropical"

    result = ProductTextNormalizer().normalize(text, source_name="sporade.jpeg")

    assert result.nombre_producto == "Sporade Tropical 500 ml"
    assert result.marca == "Sporade"
    assert result.tipo_producto == "Bebida rehidratante"
    assert result.contenido_neto == "500 ml"
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


def test_normalizer_uses_full_label_text_for_baby_dove_pack() -> None:
    text = "jabon en barra\nbabyDove\nhipoalergenico\nhumectacion sensible\n3x75.g"

    result = ProductTextNormalizer().normalize(text)

    assert result.marca == "Dove"
    assert result.tipo_producto == "Jabón"
    assert result.contenido_neto == "3 x 75 g"
    assert result.presentacion == "3 x 75 g"
    assert result.categoria_sugerida == "bebés y mamá"
    assert "Dove" in (result.nombre_producto or "")


def test_normalizer_repairs_gloria_griego_ocr_text() -> None:
    text = "GLORIA\nGRIZGO\nsabor a\nFRUTOSROJOS\nCremosidad unica\n800 g"

    result = ProductTextNormalizer().normalize(text)

    assert result.nombre_producto == "Yogurt Gloria Frutos Rojos"
    assert result.marca == "Gloria"
    assert result.tipo_producto == "Yogurt"
    assert result.contenido_neto == "800 g"
    assert result.categoria_sugerida == "lacteos"


def test_normalizer_prioritizes_prominent_voraz_brand_over_slogan() -> None:
    text = (
        "F*CKSWEET\nSTAYSALTY\nF*CK SWEET\nSTAY SALTY\nPAPASKETT\n"
        "PAPASKETTLROON\nSALDEMARAS\nPAPASKETTLEOON\nSNACKS\nVORAZ\n"
        "PAPASKETTLRCON\n1359\nSALDBMARAS\nCCYOAHRCU"
    )

    result = ProductTextNormalizer().normalize(text, source_name="voraz azul.png")

    assert result.nombre_producto == "Snack Voraz Sal de Maras"
    assert result.marca == "Voraz"
    assert result.tipo_producto == "Snack"
    assert result.contenido_neto == "135 g"
    assert result.unidad_medida == "g"
    assert result.categoria_sugerida == "snacks y golosinas"


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


def test_normalizer_detects_portugal_body_cream_from_joined_ocr_text() -> None:
    text = (
        "PORTUGAL\nCREMACORPORAL\ncoco fresh\nCONACEITEDECOCO\n"
        "Manteca de Karitey Vitamina E\nEXTRA HIDRATACION\n"
        "Ayudaameoraanvdau\ndolapel\nTODOTIPODEPIEL"
    )

    result = ProductTextNormalizer().normalize(text, source_name="portugal coco.png")

    assert result.nombre_producto == "Crema Corporal Portugal Coco Fresh"
    assert result.marca == "Portugal"
    assert result.tipo_producto == "Crema"
    assert result.categoria_sugerida == "cuidado personal"
    assert "No se identificó marca" not in " ".join(result.warnings)


def test_normalizer_repairs_nivea_body_cream_ocr_text() -> None:
    text = (
        "NIVEA\nCrema Corpora\nMilk\nNutritiva\ntr ctny.Hamectockn\n"
        "C85\nLprotundo\nDotoco\necaAmendronAMtomnoE"
    )

    result = ProductTextNormalizer().normalize(text, source_name="nivea.png")

    assert result.nombre_producto == "Crema Corporal Nivea Milk Nutritiva"
    assert result.marca == "Nivea"
    assert result.tipo_producto == "Crema"
    assert result.categoria_sugerida == "cuidado personal"
    assert "No se identificó tipo" not in " ".join(result.warnings)


def test_normalizer_detects_johnsons_soap_multipack() -> None:
    text = (
        "PACK x3JABONES\nbaby\njabón.cremoso\nconingredienteshidratantes\n"
        "pieldelicada\nlibre deparabenos\nyftalatos\n3 x 125g c/u\n"
        "hpoaleigenios\nPRCEADO\nPuedeperderhasta12.5gc/u"
    )

    result = ProductTextNormalizer().normalize(text, source_name="johnsons pack jabones.png")

    assert result.nombre_producto == "Jabón Cremoso Johnson's Bebé"
    assert result.marca == "Johnson's"
    assert result.tipo_producto == "Jabón"
    assert result.presentacion == "pack x 3"
    assert result.contenido_neto == "3 x 125 g"
    assert result.unidad_medida == "g"
    assert result.categoria_sugerida == "bebés y mamá"


def test_normalizer_detects_johnsons_brand_from_noisy_filename_and_ocr() -> None:
    text = (
        "PACK x3 JABONES\nPACKx3'JABONES\nbaby\njabón.cremoso\n"
        "coningredienteshidratantes\npieldelicada\nconingredicnteshidratantes\n"
        "librede parabenos\nyftalatos\n3x1\nPuedeperd\nPACK x3'JABONES\n"
        "libredeparabenos\nRIE\n3 x 125g c/u\nhpoeleigenics\nPACEADO\n"
        "Puedeperderhasta12.5gclu"
    )

    result = ProductTextNormalizer().normalize(text, source_name="jhonsos baby.png")

    assert result.nombre_producto == "Jabón Cremoso Johnson's Bebé"
    assert result.marca == "Johnson's"
    assert result.tipo_producto == "Jabón"
    assert result.presentacion == "pack x 3"
    assert result.contenido_neto == "3 x 125 g"
    assert result.unidad_medida == "g"
    assert result.categoria_sugerida == "bebés y mamá"


def test_normalizer_infers_soap_multipack_unit_when_ocr_drops_g() -> None:
    text = "JADON\nbabyDove\njabon en barra\nhumectacion sensible\n3x75"

    result = ProductTextNormalizer().normalize(text, source_name="dove-ba.png")

    assert result.tipo_producto == "Jabón"
    assert result.contenido_neto == "3 x 75 g"
    assert result.unidad_medida == "g"


def test_normalizer_detects_kolynos_loose_offer_pack() -> None:
    text = (
        "3\nCREMAS\nOFERTA\nESPECIAL\nDENTALES\n60mL\n"
        "Pvsprotoniedaalogrecipregufar sspendoCerniaal cubco\n"
        "CREHACENTALCOWFLCOR+CALCO\nKolynos\nSUPER\nBLANCO\n"
        "EMPAQUE\nFAMILIAR"
    )

    result = ProductTextNormalizer().normalize(text, source_name="kolinos.png")

    assert result.nombre_producto == "Pasta dental Kolynos Super Blanco"
    assert result.marca == "Kolynos"
    assert result.tipo_producto == "Pasta dental"
    assert result.presentacion == "pack x 3"
    assert result.contenido_neto == "3 x 60 ml"
    assert result.unidad_medida == "ml"
    assert result.categoria_sugerida == "cuidado personal"


def test_normalizer_detects_boreal_detergent_from_ocr_text() -> None:
    text = "boreal\nDetergente\nliquido\nAroma\nLawanda\nCo3L"

    result = ProductTextNormalizer().normalize(text, source_name="boreal.png")

    assert result.nombre_producto == "Detergente Boreal Lavanda"
    assert result.marca == "Boreal"
    assert result.tipo_producto == "Detergente"
    assert result.contenido_neto == "3 L"
    assert result.categoria_sugerida == "limpieza"


def test_normalizer_prefers_detected_bolivar_variant_over_filename_color() -> None:
    text = (
        "ABRIRAOU\nNUEVA\nIMAGEN\nDETERGENTE\nROSAS Y MAGNOLIAS\n"
        "Bolivar\nCuidadoy\nSuavidad\nSUAVIDAD Y\ncontogue\n"
        "FRAGANCIA\nde suav zante\nycapsulas\nPROLONGADA\ndearoma"
    )

    result = ProductTextNormalizer().normalize(
        text,
        source_name="bol-morado_png.rf.8deff10bb587d9bb8525ae403509c340.jpg",
    )

    assert result.nombre_producto == "Detergente Bolívar Rosas y Magnolias"
    assert result.marca == "Bolívar"
    assert result.tipo_producto == "Detergente"
    assert result.categoria_sugerida == "limpieza"


def test_normalizer_detects_bolivar_cuidado_total_variant() -> None:
    text = (
        "NUEVA\nIMAGEN\nDCJERSENTEF!CFJE\nw 4kg\nBolivar\n"
        "Cuidado\nTotal\nPROTEGE\ncontsarculas\nELCOLORY\n"
        "protictoras\nLASFIBRAS"
    )

    result = ProductTextNormalizer().normalize(text, source_name="bolivar.png")

    assert result.nombre_producto == "Jabón Bolívar Cuidado Total"
    assert result.marca == "Bolívar"
    assert result.tipo_producto == "Jabón"
    assert result.contenido_neto == "4 kg"


def test_normalizer_prioritizes_large_pharmacy_name_over_label_details() -> None:
    text = (
        "Dextrometorfano\n"
        "15 mg/5 mL\n"
        "Jarabe-Via oral\n"
        "Supresor de la tos\n"
        "120mL\n"
        "Medifarma"
    )

    result = ProductTextNormalizer().normalize(text)

    assert result.nombre_producto == "Dextrometorfano Jarabe Medifarma"
    assert result.marca == "Medifarma"
    assert result.tipo_producto == "Jarabe"
    assert result.contenido_neto == "120 ml"


def test_normalizer_ignores_side_panel_fragments_for_pharmacy_name() -> None:
    text = (
        "COMPI\n"
        "Dextrometorfano\n"
        "Dextron\n"
        "Bromn\n"
        "Excipi\n"
        "15mg/5mL\n"
        "'NOS\n"
        "NINO:\n"
        "Vent\n"
        "Jarabe-Via oral\n"
        "Ma\n"
        "Supresor de la tos\n"
        "120ml\n"
        "Medifarma"
    )

    result = ProductTextNormalizer().normalize(text, source_name="dextri.jpeg")

    assert result.nombre_producto == "Dextrometorfano Jarabe Medifarma"
    assert result.contenido_neto == "120 ml"


def test_normalizer_detects_pharmacy_cream_with_percentage_concentration() -> None:
    text = (
        "AC\n"
        "BETAMETASONA\n"
        "0,05%\n"
        "Linea(DCI\n"
        "CREMA\n"
        "LABORATORIOSACFARMA S.A.\n"
        "LABORATORIOSAC FARMA S.A"
    )

    result = ProductTextNormalizer().normalize(text, source_name="betametasona.jpeg")

    assert result.nombre_producto == "Betametasona Crema AC Farma"
    assert result.marca == "AC Farma"
    assert result.tipo_producto == "Crema"
    assert result.presentacion == "crema 0.05 %"
    assert result.contenido_neto == "0.05 %"
    assert result.unidad_medida == "%"
    assert result.categoria_sugerida == "farmacia/otc"


def test_normalizer_prefers_prominent_ocr_line_as_general_product_name_rule() -> None:
    text = (
        "ANTIACIDO\n"
        "GASEOVET MS\n"
        "Magaldrato + Simeticona\n"
        "(800 mg + 60 mg)/10 mL\n"
        "Suspensión Oral\n"
        "Vía oral\n"
        "220 mL"
    )

    result = ProductTextNormalizer().normalize(text, prominent_text="GASEOVET MS")

    assert result.nombre_producto == "Gaseovet Ms 220 ml"
    assert result.categoria_sugerida == "farmacia/otc"
    assert result.contenido_neto == "220 ml"


def test_normalizer_detects_cartavio_ron_as_beverage() -> None:
    text = (
        "RON CARTAVIO BLACK BARREL 3ANOS BARRICAS\n"
        "INTENSAMENTE TOSTADO\n"
        "ANEJADO EN B AHUMADO"
    )

    result = ProductTextNormalizer().normalize(text, prominent_text="Cartavio")

    assert result.marca == "Cartavio"
    assert result.tipo_producto == "Ron"
    assert result.categoria_sugerida == "bebidas"


def test_normalizer_always_returns_category_for_unknown_product() -> None:
    result = ProductTextNormalizer().normalize("Producto sin senales claras")

    assert result.categoria_sugerida == "otros"
