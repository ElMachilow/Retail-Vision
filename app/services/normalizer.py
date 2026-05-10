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
    pack_regex = re.compile(r"\b(?:pack|paq|caja)?\s*x\s*(?P<count>\d{1,3})\b", flags=re.IGNORECASE)

    def normalize(self, text: str, source_name: str | None = None) -> NormalizedProduct:
        clean_text = self._clean_text(text)
        clean_source_name = self._clean_source_name(source_name)
        clean_context = "\n".join(part for part in (clean_text, clean_source_name) if part)
        folded_text = self._fold(clean_text)
        folded_context = self._fold(clean_context)
        warnings: list[str] = []

        brand, category_hint = self._detect_brand(folded_context)
        product_type = self._detect_product_type(folded_context, brand)
        amount, unit = self._detect_content(clean_context)
        category = category_hint or self._detect_category(folded_context) or self._category_from_product_type(product_type)
        presentation = self._detect_presentation(clean_text, amount, unit)
        variant = self._detect_variant(folded_context, category)
        name = self._build_product_name(clean_text, brand, amount, unit, variant, category, product_type)

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

        content = f"{amount} {unit}" if amount and unit else None
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
        return self._remove_warning_label_text(text.strip())

    def _remove_warning_label_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        cleaned: list[str] = []
        skip_warning_parts = 0
        for line in lines:
            if not line:
                continue

            folded_line = self._fold(line)
            if self._is_warning_label_line(folded_line):
                skip_warning_parts = 2 if re.search(r"\balto\s+en\b", folded_line) else skip_warning_parts
                continue

            if skip_warning_parts > 0 and self._is_warning_label_fragment(folded_line):
                skip_warning_parts -= 1
                continue

            skip_warning_parts = 0
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _is_warning_label_line(self, folded_line: str) -> bool:
        return bool(
            re.search(r"\balto\s+en\b", folded_line)
            or "evitar su consumo" in folded_line
            or "evitar consumo" in folded_line
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
        matches = list(self.content_regex.finditer(clean_text))
        if not matches:
            return None, None

        def score(match: re.Match) -> tuple[int, int]:
            raw_unit = match.group("unit").lower().replace(".", "")
            normalized_unit = UNIT_ALIASES.get(raw_unit, raw_unit)
            priority = 0 if normalized_unit in {"g", "kg", "ml", "L"} else 1
            return (priority, match.start())

        selected = sorted(matches, key=score)[0]
        amount = selected.group("amount").replace(",", ".")
        unit_key = selected.group("unit").lower().replace(".", "")
        return amount, UNIT_ALIASES.get(unit_key, unit_key)

    def _detect_category(self, folded_text: str) -> str | None:
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                folded_keyword = self._fold(keyword)
                if re.search(rf"(?<!\w){re.escape(folded_keyword)}(?!\w)", folded_text):
                    return category
        return None

    def _detect_product_type(self, folded_text: str, brand: str | None) -> str | None:
        padded = f" {folded_text} "
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
        for word in ("sachet", "botella", "frasco", "sobre", "caja", "pack"):
            if re.search(rf"\b{word}s?\b", lowered):
                if amount and unit:
                    return f"{word} {amount} {unit}"
                return word

        pack_match = self.pack_regex.search(clean_text)
        if pack_match:
            return f"pack x {pack_match.group('count')}"

        if amount and unit:
            return f"{amount} {unit}"
        return None

    def _build_product_name(
        self,
        clean_text: str,
        brand: str | None,
        amount: str | None,
        unit: str | None,
        variant: str | None,
        category: str | None,
        product_type: str | None,
    ) -> str | None:
        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        meaningful = [line for line in lines if self._is_meaningful_line(line)]
        if product_type:
            return self._compose_product_name(product_type, brand, variant)

        if brand:
            brand_name = self._compose_brand_name(brand, variant)
            if brand_name:
                product_line = self._find_product_descriptor(meaningful, brand, variant, category)
                if product_line and self._fold(product_line) not in self._fold(brand_name):
                    return f"{brand_name} {product_line}"[:120]
                return brand_name[:120]

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

    def _compose_product_name(
        self,
        product_type: str,
        brand: str | None,
        variant: str | None,
    ) -> str:
        parts = [product_type]
        if brand and self._fold(brand) not in self._fold(product_type):
            parts.append(brand)

        variant_suffix = self._variant_suffix(product_type, brand, variant)
        if variant_suffix:
            parts.append(variant_suffix)

        return " ".join(parts)[:120]

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
            "snacks y golosinas": ("galleta", "chocolate", "snack"),
            "limpieza": ("detergente", "lejia", "lejía", "lavavajilla"),
            "cuidado personal": ("shampoo", "pasta dental", "crema", "jabon", "jabón"),
            "farmacia/otc": ("tableta", "pastilla", "alcohol", "jarabe"),
        }.get(category or "", ())

        for line in meaningful:
            folded_line = self._fold(line)
            if folded_brand in folded_line or (folded_variant and folded_variant in folded_line):
                continue
            if any(self._fold(keyword) in folded_line for keyword in product_keywords):
                return self.content_regex.sub("", line).strip(" -.,")
        return None

    def _is_meaningful_line(self, line: str) -> bool:
        folded_line = self._fold(line)
        if len(folded_line) < 3:
            return False
        if any(phrase in folded_line for phrase in (self._fold(item) for item in NOISE_PHRASES)):
            return False
        tokens = set(re.findall(r"[a-z0-9]+", folded_line))
        if tokens and tokens.issubset({self._fold(stopword) for stopword in PRODUCT_STOPWORDS}):
            return False
        if re.fullmatch(r"[\d\W]+", folded_line):
            return False
        return True
