import re
import unicodedata
from dataclasses import dataclass

from app.domain.catalog import (
    BRAND_ALIASES,
    BRAND_DEFAULT_PRODUCT_TYPES,
    CATEGORY_KEYWORDS,
    NOISE_PHRASES,
    PRODUCT_STOPWORDS,
    PRODUCT_TYPE_ALIASES,
    PRODUCT_TYPE_CATEGORIES,
    UNIT_ALIASES,
    VARIANT_ALIASES,
)


@dataclass(frozen=True)
class NormalizedProduct:
    nombre_producto: str | None
    marca: str | None
    tipo_producto: str | None
    presentacion: str | None
    contenido_neto: str | None
    unidad_medida: str | None
    categoria_sugerida: str | None
    warnings: list[str]


class ProductTextNormalizer:
    unit_pattern = "|".join(sorted((re.escape(unit) for unit in UNIT_ALIASES), key=len, reverse=True))
    content_regex = re.compile(
        rf"(?P<amount>\d+(?:[.,]\d+)?)\s*(?P<unit>{unit_pattern})\b",
        flags=re.IGNORECASE,
    )
    concentration_regex = re.compile(
        r"(?P<amount>\d+(?:[.,]\d+)?)\s*(?P<unit>%)",
        flags=re.IGNORECASE,
    )
    multipack_content_regex = re.compile(
        rf"(?P<count>\d{{1,3}})\s*x\s*(?P<amount>\d+(?:[.,]\d+)?)\s*(?P<unit>{unit_pattern})\b(?:\s*c\s*/?\s*u)?",
        flags=re.IGNORECASE,
    )
    soap_multipack_missing_unit_regex = re.compile(
        r"(?P<count>\d{1,3})\s*x\s*(?P<amount>75|90|100|110|125|150)(?=\D|$)",
        flags=re.IGNORECASE,
    )
    loose_pack_count_regex = re.compile(r"(?m)^\s*(?P<count>[2-9]|[1-9]\d)\s*$")
    ocr_g_content_regex = re.compile(r"(?<!\d)(?P<amount>\d{1,3})[9q](?!\d)", flags=re.IGNORECASE)
    pack_regex = re.compile(r"\b(?:pack|paq|caja)?\s*x\s*(?P<count>\d{1,3})\b", flags=re.IGNORECASE)
    diaper_count_regex = re.compile(
        r"(?P<count>\d{1,3})\s*(?:panales|pañales|panal|pañal)\b",
        flags=re.IGNORECASE,
    )

    pharmacy_detail_words = {
        "administracion",
        "bpm",
        "bromn",
        "compi",
        "composicion",
        "contiene",
        "dosis",
        "dextron",
        "excipi",
        "excipientes",
        "formula",
        "iso",
        "laboratorio",
        "laboratorios",
        "ma",
        "mane",
        "nino",
        "nos",
        "oral",
        "para",
        "proley",
        "registro",
        "supresor",
        "suspension",
        "tos",
        "vent",
        "ventas",
        "via",
    }
    active_ingredient_words = {
        "betametasona",
        "clotrimazol",
        "dexametasona",
        "hidrocortisona",
        "mupirocina",
    }

    def normalize(
        self,
        text: str,
        source_name: str | None = None,
        prominent_text: str | None = None,
    ) -> NormalizedProduct:
        clean_text = self._clean_text(text)
        clean_prominent_text = self._clean_text(prominent_text or "")
        clean_source_name = self._clean_source_name(source_name)
        clean_context = "\n".join(part for part in (clean_text, clean_source_name) if part)
        folded_text = self._fold(clean_text)
        folded_context = self._fold(clean_context)
        warnings: list[str] = []

        brand, category_hint = self._detect_brand(folded_context)
        product_type = self._detect_product_type(folded_context, brand)
        amount, unit = self._detect_content(clean_context)
        detected_category = self._detect_category(folded_context)
        category = self._category_from_context(product_type, brand, folded_context) or self._category_from_product_type(product_type) or detected_category or category_hint
        presentation = self._detect_presentation(clean_text, amount, unit)
        variant = self._detect_variant(folded_context, category)
        content = f"{amount} {unit}" if amount and unit else None
        name = self._build_product_name(
            clean_text,
            brand,
            amount,
            unit,
            variant,
            category,
            product_type,
            content,
            prominent_text=clean_prominent_text,
        )

        if not text.strip():
            warnings.append("OCR no devolvió texto; se requiere revisión manual.")
        if brand is None:
            warnings.append("No se identificó marca en el catálogo peruano inicial.")
        if product_type is None:
            warnings.append("No se identificó tipo de producto con reglas de dominio.")
        if amount is None or unit is None:
            warnings.append("No se detectó contenido neto o unidad de medida con confianza.")
        if category is None:
            warnings.append("No se pudo sugerir categoría con reglas de dominio.")

        return NormalizedProduct(
            nombre_producto=name,
            marca=brand,
            tipo_producto=product_type,
            presentacion=presentation,
            contenido_neto=content,
            unidad_medida=unit,
            categoria_sugerida=category,
            warnings=warnings,
        )

    def _clean_text(self, text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"[|_]+", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = self._repair_common_ocr_tokens(text)
        return self._remove_warning_label_text(text.strip())

    def _repair_common_ocr_tokens(self, text: str) -> str:
        # OCR often reads small "g" units as 9/q on retail packs, e.g. "42g" -> "429".
        text = self.ocr_g_content_regex.sub(r"\g<amount> g", text)
        text = re.sub(r"\bbaby\s*dove\b", "Baby Dove", text, flags=re.IGNORECASE)
        text = re.sub(r"\bbabydove\b", "Baby Dove", text, flags=re.IGNORECASE)
        text = re.sub(r"\bfrutos\s*rojos\b", "Frutos Rojos", text, flags=re.IGNORECASE)
        text = re.sub(r"\bGRIZGO\b", "GRIEGO", text, flags=re.IGNORECASE)
        text = re.sub(r"\bJADON\b", "JABÓN", text, flags=re.IGNORECASE)
        text = re.sub(r"\bJADONES\b", "JABONES", text, flags=re.IGNORECASE)
        text = re.sub(r"\bPACK\s*x\s*(\d+)\s*['’]\s*JABONES\b", r"PACK x\1 JABONES", text, flags=re.IGNORECASE)
        text = re.sub(r"\bPACKx\s*(\d+)\s*['’]?\s*JABONES\b", r"PACK x\1 JABONES", text, flags=re.IGNORECASE)
        text = re.sub(r"\bjabon\s*en\s*barra\b", "jabon en barra", text, flags=re.IGNORECASE)
        text = re.sub(r"\bjabonen\s*barra\b", "jabon en barra", text, flags=re.IGNORECASE)
        text = re.sub(r"\bjab[oó]n[.\s]*cremoso\b", "jabón cremoso", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcon\s*ingredic?ntes\s*hidratantes\b", "con ingredientes hidratantes", text, flags=re.IGNORECASE)
        text = re.sub(r"\blibre\s*de\s*parabenos\b", "libre de parabenos", text, flags=re.IGNORECASE)
        text = re.sub(r"\bpiel\s*delicada\b", "piel delicada", text, flags=re.IGNORECASE)
        text = re.sub(r"\bPAPASKETT(?:LE)?(?:LR|L)?(?:OON|CON)?\b", "PAPAS KETTLE CON", text, flags=re.IGNORECASE)
        text = re.sub(r"\bSALD[BE]?MARAS\b", "SAL DE MARAS", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<=\d)\.\s*(?=g\b|kg\b|ml\b|l\b)", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bALTOEN\b", "ALTO EN", text, flags=re.IGNORECASE)
        text = re.sub(r"\bCREMACORPORAL\b", "CREMA CORPORAL", text, flags=re.IGNORECASE)
        text = re.sub(r"\bCREMA\s+CORPORA\b", "CREMA CORPORAL", text, flags=re.IGNORECASE)
        text = re.sub(r"\bCONACEITEDECOCO\b", "CON ACEITE DE COCO", text, flags=re.IGNORECASE)
        text = re.sub(r"\bTODOTIPODEPIEL\b", "TODO TIPO DE PIEL", text, flags=re.IGNORECASE)
        text = re.sub(r"\bPACK\s*x\s*(\d+)(?=[A-ZÁÉÍÓÚÜÑ])", r"PACK x \1 ", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<=\d)\s*g\s*c\s*/?\s*u\b", " g c/u", text, flags=re.IGNORECASE)
        return text

    def _remove_warning_label_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        cleaned: list[str] = []
        skip_warning_parts = 0
        for line in lines:
            if not line:
                continue

            folded_line = self._fold(line)
            if self._is_warning_label_line(folded_line):
                skip_warning_parts = 4 if re.search(r"\balto\s*en\b", folded_line) else skip_warning_parts
                continue

            if skip_warning_parts > 0 and self._is_warning_label_fragment(folded_line):
                skip_warning_parts -= 1
                continue

            skip_warning_parts = 0
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _is_warning_label_line(self, folded_line: str) -> bool:
        return bool(
            re.search(r"\balto\s*en\b", folded_line)
            or "evitar su consumo" in folded_line
            or "evitar consumo" in folded_line
            or self._looks_like_evitar_consumo_noise(folded_line)
        )

    def _is_warning_label_fragment(self, folded_line: str) -> bool:
        return folded_line in {
            "azucar",
            "sodio",
            "grasas",
            "grasas saturadas",
            "saturadas",
            "alto",
            "en",
        }

    def _looks_like_evitar_consumo_noise(self, folded_line: str) -> bool:
        compact = re.sub(r"[^a-z]", "", folded_line)
        return bool(compact and ("consu" in compact or "cisvo" in compact) and len(compact) <= 24)

    def _clean_source_name(self, source_name: str | None) -> str:
        if not source_name:
            return ""
        source_name = re.sub(r"\.[a-zA-Z0-9]{2,5}$", "", source_name)
        source_name = re.sub(r"[_\-]+", " ", source_name)
        source_name = re.sub(r"\b\d{2,}\b", " ", source_name)
        return self._clean_text(source_name)

    def _fold(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower())
        ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
        return re.sub(r"\s+", " ", ascii_text)

    def _detect_brand(self, folded_text: str) -> tuple[str | None, str | None]:
        padded = f" {folded_text} "
        for brand in BRAND_ALIASES:
            for alias in brand.aliases:
                folded_alias = self._fold(alias)
                if re.search(rf"(?<!\w){re.escape(folded_alias)}(?!\w)", padded):
                    return brand.canonical, brand.category_hint
        return None, None

    def _detect_content(self, clean_text: str) -> tuple[str | None, str | None]:
        diaper_count_match = self.diaper_count_regex.search(clean_text)
        if diaper_count_match:
            return diaper_count_match.group("count"), "unid"

        multipack_match = self.multipack_content_regex.search(clean_text)
        if multipack_match:
            count = multipack_match.group("count")
            amount = multipack_match.group("amount").replace(",", ".")
            unit_key = multipack_match.group("unit").lower().replace(".", "")
            return f"{count} x {amount}", UNIT_ALIASES.get(unit_key, unit_key)

        soap_multipack_match = self.soap_multipack_missing_unit_regex.search(clean_text)
        if soap_multipack_match and self._looks_like_soap_pack(clean_text):
            count = soap_multipack_match.group("count")
            amount = soap_multipack_match.group("amount")
            return f"{count} x {amount}", "g"

        loose_pack = self._detect_loose_multipack_content(clean_text)
        if loose_pack:
            return loose_pack

        matches = list(self.content_regex.finditer(clean_text))
        if not matches:
            concentration_match = self.concentration_regex.search(clean_text)
            if concentration_match and self._looks_like_pharmacy_context(self._fold(clean_text)):
                return concentration_match.group("amount").replace(",", "."), "%"
            return None, None

        def score(match: re.Match) -> tuple[int, int, int]:
            raw_unit = match.group("unit").lower().replace(".", "")
            normalized_unit = UNIT_ALIASES.get(raw_unit, raw_unit)
            dosage_penalty = 1 if self._looks_like_dosage_content(clean_text, match) else 0
            priority = 0 if normalized_unit in {"g", "kg", "ml", "L"} else 1
            return (dosage_penalty, priority, match.start())

        selected = sorted(matches, key=score)[0]
        amount = selected.group("amount").replace(",", ".")
        unit_key = selected.group("unit").lower().replace(".", "")
        return amount, UNIT_ALIASES.get(unit_key, unit_key)

    def _looks_like_dosage_content(self, clean_text: str, match: re.Match) -> bool:
        start, end = match.span()
        before = clean_text[max(0, start - 2) : start]
        after = clean_text[end : end + 2]
        if "/" in before or "/" in after:
            return True

        line_start = clean_text.rfind("\n", 0, start) + 1
        line_end = clean_text.find("\n", end)
        if line_end == -1:
            line_end = len(clean_text)
        folded_line = self._fold(clean_text[line_start:line_end])
        return bool(
            "/" in folded_line
            or (
                re.search(r"\b(dosis|mcg|mg|ui|iu)\b", folded_line)
                and re.search(r"\b(mcg|mg|ui|iu)\b", folded_line)
            )
        )

    def _detect_loose_multipack_content(self, clean_text: str) -> tuple[str, str] | None:
        folded_text = self._fold(clean_text)
        has_pack_context = (
            ("cremas" in folded_text and "dentales" in folded_text)
            or "jabones" in folded_text
            or "unidades" in folded_text
            or "pack" in folded_text
        )
        if not has_pack_context:
            return None

        count_match = self.loose_pack_count_regex.search(clean_text)
        content_match = self.content_regex.search(clean_text)
        if not count_match or not content_match:
            return None

        amount = content_match.group("amount").replace(",", ".")
        unit_key = content_match.group("unit").lower().replace(".", "")
        return f"{count_match.group('count')} x {amount}", UNIT_ALIASES.get(unit_key, unit_key)

    def _detect_category(self, folded_text: str) -> str | None:
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                folded_keyword = self._fold(keyword)
                if re.search(rf"(?<!\w){re.escape(folded_keyword)}(?!\w)", folded_text):
                    return category
        return None

    def _detect_product_type(self, folded_text: str, brand: str | None) -> str | None:
        padded = f" {folded_text} "
        if re.search(r"(?<!\w)crema\s*corporal(?!\w)", padded):
            return "Crema"
        for product_type, aliases in PRODUCT_TYPE_ALIASES.items():
            for alias in aliases:
                folded_alias = self._fold(alias)
                if re.search(rf"(?<!\w){re.escape(folded_alias)}(?!\w)", padded):
                    return product_type
        if brand:
            return BRAND_DEFAULT_PRODUCT_TYPES.get(brand)
        return None

    def _category_from_product_type(self, product_type: str | None) -> str | None:
        if not product_type:
            return None
        return PRODUCT_TYPE_CATEGORIES.get(product_type)

    def _category_from_context(
        self,
        product_type: str | None,
        brand: str | None,
        folded_text: str,
    ) -> str | None:
        if product_type == "Jabón" and (
            brand == "Johnson's"
            or re.search(r"(?<!\w)(baby|bebe|bebe|piel delicada)(?!\w)", folded_text)
            or re.search(r"(?<!\w)baby\s*dove(?!\w)", folded_text)
        ):
            return "bebés y mamá"
        if product_type == "Jabón" and re.search(r"(?<!\w)(piel|humectacion|hipoalergenico)(?!\w)", folded_text):
            return "cuidado personal"
        if product_type == "Crema" and self._looks_like_pharmacy_context(folded_text):
            return "farmacia/otc"
        return None

    def _looks_like_pharmacy_context(self, folded_text: str) -> bool:
        tokens = set(re.findall(r"[a-z0-9]+", folded_text))
        return bool(
            tokens & self.active_ingredient_words
            or re.search(r"\b(laboratorio|laboratorios|farma|dci|via oral|registro sanitario)\b", folded_text)
            or re.search(r"\b\d+(?:[.,]\d+)?\s*%", folded_text)
        )

    def _detect_variant(self, folded_text: str, category: str | None) -> str | None:
        for canonical, aliases in VARIANT_ALIASES.items():
            for alias in aliases:
                folded_alias = self._fold(alias)
                if re.search(rf"(?<!\w){re.escape(folded_alias)}(?!\w)", folded_text):
                    if canonical in {"Leche Entera", "Leche Evaporada"} and category not in {"lacteos", None}:
                        continue
                    return canonical
        return None

    def _detect_presentation(self, clean_text: str, amount: str | None, unit: str | None) -> str | None:
        lowered = clean_text.lower()
        pack_match = self.pack_regex.search(clean_text)
        if pack_match:
            return f"pack x {pack_match.group('count')}"
        loose_pack_match = self.loose_pack_count_regex.search(clean_text)
        if loose_pack_match and self._looks_like_pack_offer(clean_text):
            return f"pack x {loose_pack_match.group('count')}"
        diaper_count_match = self.diaper_count_regex.search(clean_text)
        if diaper_count_match:
            return f"{diaper_count_match.group('count')} pañales"

        for word in ("sachet", "botella", "frasco", "sobre", "caja", "pack", "tubo", "crema", "pomada"):
            if re.search(rf"\b{word}s?\b", lowered):
                if amount and unit:
                    return f"{word} {amount} {unit}"
                return word

        if amount and unit:
            return f"{amount} {unit}"
        return None

    def _looks_like_pack_offer(self, clean_text: str) -> bool:
        folded_text = self._fold(clean_text)
        return any(
            marker in folded_text
            for marker in ("cremas dentales", "jabones", "oferta especial", "pack")
        )

    def _looks_like_soap_pack(self, clean_text: str) -> bool:
        folded_text = self._fold(clean_text)
        return "jabon" in folded_text or "jabones" in folded_text

    def _build_product_name(
        self,
        clean_text: str,
        brand: str | None,
        amount: str | None,
        unit: str | None,
        variant: str | None,
        category: str | None,
        product_type: str | None,
        content: str | None,
        prominent_text: str | None = None,
    ) -> str | None:
        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        meaningful = [line for line in lines if self._is_meaningful_line(line)]
        prominent_line = self._clean_prominent_product_line(prominent_text, brand, product_type)
        if prominent_line and self._should_prefer_prominent_line(prominent_line, brand, product_type, category):
            return self._compose_name_from_prominent_line(prominent_line, brand, product_type, content)[:120]

        if category == "farmacia/otc":
            primary_line = self._find_primary_pharmacy_line(meaningful, brand, product_type)
            if primary_line:
                return self._compose_pharmacy_name(primary_line, brand, product_type)

        if brand:
            brand_name = self._compose_brand_name(brand, variant)
            if brand_name:
                product_line = self._find_product_descriptor(meaningful, brand, variant, category)
                if product_line and self._fold(product_line) not in self._fold(brand_name):
                    return self._compose_descriptor_name(product_line, product_type, brand_name, variant)
                if not product_type:
                    return brand_name[:120]

        if product_type:
            return self._compose_product_name(
                self._display_product_type(product_type, clean_text),
                brand,
                variant,
                content,
            )

        if not meaningful:
            return brand

        selected = None
        selected = selected or meaningful[0]

        selected = self.content_regex.sub("", selected)
        selected = re.sub(r"\b\d{5,}\b", "", selected)
        selected = re.sub(r"\s{2,}", " ", selected).strip(" -.,")
        if selected:
            return selected[:120]
        return brand or (f"Producto {amount} {unit}" if amount and unit else None)

    def _clean_prominent_product_line(
        self,
        prominent_text: str | None,
        brand: str | None,
        product_type: str | None,
    ) -> str | None:
        if not prominent_text:
            return None
        line = self.content_regex.sub("", prominent_text)
        line = self.concentration_regex.sub("", line)
        line = re.sub(r"[®™©]+", "", line)
        line = re.sub(r"\b\d{5,}\b", "", line)
        line = re.sub(r"\s{2,}", " ", line).strip(" -.,:;")
        if not line or not self._is_meaningful_line(line):
            return None
        folded_line = self._fold(line)
        if self._is_pharmacy_detail_line(folded_line):
            return None
        if self._fold(line) in {self._fold(brand or ""), self._fold(product_type or "")}:
            return None
        return line

    def _should_prefer_prominent_line(
        self,
        prominent_line: str,
        brand: str | None,
        product_type: str | None,
        category: str | None,
    ) -> bool:
        folded_line = self._fold(prominent_line)
        if category == "farmacia/otc":
            return True
        if len(re.findall(r"[a-z0-9]+", folded_line)) >= 2:
            return True
        if not brand and not product_type:
            return True
        return False

    def _compose_name_from_prominent_line(
        self,
        prominent_line: str,
        brand: str | None,
        product_type: str | None,
        content: str | None = None,
    ) -> str:
        parts = [self._title_product_phrase(prominent_line)]
        folded_parts = self._fold(" ".join(parts))
        drug_ingredient_types = {
            "Sulfato Ferroso",
            "Simeticona",
            "Clopidogrel",
            "Loratadina",
            "Omeprazol",
            "Metformina",
            "Antiácido",
        }
        prominent_token_count = len(re.findall(r"[a-z0-9]+", self._fold(prominent_line)))
        if (
            product_type
            and prominent_token_count < 2
            and product_type not in drug_ingredient_types
            and self._fold(product_type) not in folded_parts
        ):
            parts.append(product_type)
        folded_parts = self._fold(" ".join(parts))
        if brand and self._fold(brand) not in folded_parts:
            parts.append(brand)
        folded_parts = self._fold(" ".join(parts))
        if content and self._fold(content) not in folded_parts:
            parts.append(content)
        return " ".join(parts)

    def _find_primary_pharmacy_line(
        self,
        meaningful: list[str],
        brand: str | None,
        product_type: str | None,
    ) -> str | None:
        folded_brand = self._fold(brand or "")
        folded_type = self._fold(product_type or "")
        candidates: list[tuple[int, int, str]] = []
        for index, line in enumerate(meaningful):
            cleaned = self.content_regex.sub("", line).strip(" -.,")
            folded_line = self._fold(cleaned)
            if not folded_line:
                continue
            if folded_brand and folded_brand in folded_line:
                continue
            if folded_type and folded_type in folded_line:
                continue
            if self._is_pharmacy_detail_line(folded_line):
                continue
            candidates.append((self._score_primary_pharmacy_line(folded_line, index), -index, cleaned))
        if not candidates:
            return None
        return max(candidates)[2]

    def _is_pharmacy_detail_line(self, folded_line: str) -> bool:
        tokens = set(re.findall(r"[a-z0-9]+", folded_line))
        if not tokens:
            return True
        if tokens & self.pharmacy_detail_words:
            return True
        if re.search(r"\b\d+\s*(mg|mcg|g|ml|dosis|%)\b", folded_line):
            return True
        if re.fullmatch(r"[a-z]{1,3}\d*", folded_line):
            return True
        return False

    def _score_primary_pharmacy_line(self, folded_line: str, index: int) -> int:
        tokens = re.findall(r"[a-z0-9]+", folded_line)
        alpha_len = sum(len(token) for token in tokens if re.search(r"[a-z]", token))
        score = min(alpha_len, 24)
        if len(tokens) <= 2 and alpha_len >= 8:
            score += 18
        if alpha_len < 7:
            score -= 18
        if re.search(r"\b(clorhidrato|furoato|policresuleno|excipiente|excipientes)\b", folded_line):
            score -= 10
        score -= min(index, 8)
        return score

    def _compose_pharmacy_name(
        self,
        primary_line: str,
        brand: str | None,
        product_type: str | None,
    ) -> str:
        parts = [self._title_product_phrase(primary_line)]
        if product_type and self._fold(product_type) not in self._fold(" ".join(parts)):
            parts.append(product_type)
        if brand and self._fold(brand) not in self._fold(" ".join(parts)):
            parts.append(brand)
        return " ".join(parts)[:120]

    def _compose_product_name(
        self,
        product_type: str,
        brand: str | None,
        variant: str | None,
        content: str | None = None,
    ) -> str:
        if brand and product_type in {"Bebida rehidratante", "Energizante"}:
            parts = [brand]
            variant_suffix = self._variant_suffix(product_type, brand, variant)
            if variant_suffix:
                parts.append(variant_suffix)
            if content:
                parts.append(content)
            return " ".join(parts)[:120]

        parts = [product_type]
        if brand and self._fold(brand) not in self._fold(product_type):
            parts.append(brand)

        variant_suffix = self._variant_suffix(product_type, brand, variant)
        if variant_suffix:
            parts.append(variant_suffix)

        return " ".join(parts)[:120]

    def _compose_descriptor_name(
        self,
        product_line: str,
        product_type: str | None,
        brand_name: str,
        variant: str | None,
    ) -> str:
        folded_line = self._fold(product_line)
        descriptor = product_type if product_type and folded_line in {"snack", "snacks"} else self._title_product_phrase(product_line)
        parts = [descriptor]
        if product_type and self._fold(product_type) not in folded_line:
            parts.append(product_type)
        if self._fold(brand_name) not in folded_line:
            parts.append(brand_name)
        if variant:
            folded_parts = self._fold(" ".join(parts))
            if self._fold(variant) not in folded_parts:
                parts.append(variant)
        return " ".join(parts)[:120]

    def _display_product_type(self, product_type: str, clean_text: str) -> str:
        if product_type == "Crema" and "crema corporal" in self._fold(clean_text):
            return "Crema Corporal"
        return product_type

    def _title_product_phrase(self, phrase: str) -> str:
        words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", phrase)
        if not words:
            return phrase.strip()
        return " ".join(word.capitalize() for word in words)

    def _variant_suffix(
        self,
        product_type: str,
        brand: str | None,
        variant: str | None,
    ) -> str | None:
        if not variant:
            return None

        folded_variant = self._fold(variant)
        folded_brand = self._fold(brand or "")
        if folded_variant in self._fold(product_type) or folded_variant in folded_brand:
            return None
        if product_type == "Gaseosa" and variant == "Cola":
            return None

        folded_product_type = self._fold(product_type)
        if folded_variant.startswith(folded_product_type):
            suffix = variant[len(product_type) :].strip()
            return suffix or None
        return variant

    def _compose_brand_name(self, brand: str, variant: str | None) -> str:
        if not variant:
            return brand
        if self._fold(variant) in self._fold(brand):
            return brand
        return f"{brand} {variant}"

    def _find_product_descriptor(
        self,
        meaningful: list[str],
        brand: str,
        variant: str | None,
        category: str | None,
    ) -> str | None:
        folded_brand = self._fold(brand)
        folded_variant = self._fold(variant or "")
        product_keywords = {
            "lacteos": ("leche", "yogurt", "queso"),
            "bebidas": ("gaseosa", "agua", "jugo", "nectar", "néctar"),
            "snacks y golosinas": ("galleta", "chocolate", "snack", "morochas"),
            "limpieza": ("detergente", "lejia", "lejía", "lavavajilla"),
            "cuidado personal": ("shampoo", "pasta dental", "crema", "crema corporal", "jabon", "jabón"),
            "bebés y mamá": ("jabon", "jabón", "shampoo", "crema", "panal", "pañal"),
            "farmacia/otc": ("tableta", "pastilla", "alcohol", "jarabe"),
        }.get(category or "", ())

        for line in meaningful:
            folded_line = self._fold(line)
            if folded_brand in folded_line or (folded_variant and folded_variant in folded_line):
                continue
            if self._is_pack_label_line(folded_line):
                continue
            if any(self._fold(keyword) in folded_line for keyword in product_keywords):
                return self.content_regex.sub("", line).strip(" -.,")
        return None

    def _is_pack_label_line(self, folded_line: str) -> bool:
        if re.search(r"\b(pack|paq|caja)\b", folded_line) and re.search(r"\bx\s*\d{1,3}\b", folded_line):
            return True
        if re.fullmatch(r"\d{1,3}\s*(panales|panal|panales|panal)", folded_line):
            return True
        return folded_line in {"cremas", "dentales", "oferta", "especial", "empaque", "familiar"}

    def _is_meaningful_line(self, line: str) -> bool:
        folded_line = self._fold(line)
        if len(folded_line) < 3:
            return False
        if any(phrase in folded_line for phrase in (self._fold(item) for item in NOISE_PHRASES)):
            return False
        if re.search(r"f\s*ck\s*sweet|stay\s*salty", folded_line):
            return False
        tokens = set(re.findall(r"[a-z0-9]+", folded_line))
        if tokens and tokens.issubset({self._fold(stopword) for stopword in PRODUCT_STOPWORDS}):
            return False
        if re.fullmatch(r"[\d\W]+", folded_line):
            return False
        return True
