import unicodedata
from dataclasses import dataclass
import re

from app.domain.catalog import (
    BRAND_ALIASES,
    BRAND_DEFAULT_PRODUCT_TYPES,
    PRODUCT_TYPE_ALIASES,
    PRODUCT_TYPE_CATEGORIES,
)
from app.repositories.products import ProductRepository, ProductRecord
from app.services.normalizer import ProductTextNormalizer


@dataclass(frozen=True)
class ProductSuggestionItem:
    nombre_producto: str
    marca: str | None = None
    tipo_producto: str | None = None
    categoria_sugerida: str | None = None
    source: str = "catalog"
    product_id: int | None = None


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char)).strip()
    return re.sub(r"\s+", " ", ascii_text)


def _append_unique(names: list[str], name: str | None) -> None:
    if not name:
        return
    clean_name = " ".join(name.split())
    if clean_name and _fold(clean_name) not in {_fold(item) for item in names}:
        names.append(clean_name)


class ProductSuggestionService:
    def __init__(self, repository: ProductRepository) -> None:
        self.repository = repository
        self.normalizer = ProductTextNormalizer()

    def suggest(
        self,
        query: str,
        limit: int = 3,
        context_text: str | None = None,
        source_name: str | None = None,
        prominent_text: str | None = None,
    ) -> list[ProductSuggestionItem]:
        query = (query or "").strip()
        if len(query) < 3:
            return []

        suggestions: list[ProductSuggestionItem] = []
        seen: set[str] = set()

        for item in self._context_candidates(query, context_text, source_name, prominent_text):
            key = _fold(item.nombre_producto)
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(item)
            if len(suggestions) >= limit:
                return suggestions

        for record in self.repository.search_by_name(query, limit=limit):
            key = _fold(record.nombre_producto)
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(
                ProductSuggestionItem(
                    nombre_producto=record.nombre_producto,
                    marca=record.marca,
                    tipo_producto=record.tipo_producto,
                    categoria_sugerida=record.categoria_sugerida,
                    source="db",
                    product_id=record.id,
                )
            )
            if len(suggestions) >= limit:
                return suggestions

        for item in self._catalog_candidates(query):
            key = _fold(item.nombre_producto)
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(item)
            if len(suggestions) >= limit:
                break

        return suggestions

    def _context_candidates(
        self,
        query: str,
        context_text: str | None,
        source_name: str | None,
        prominent_text: str | None,
    ) -> list[ProductSuggestionItem]:
        if not context_text:
            return []

        normalized = self.normalizer.normalize(context_text, source_name=source_name)
        inferred_prominent_text = prominent_text
        if not inferred_prominent_text and self._should_infer_prominent_for_normalized(normalized):
            inferred_prominent_text = self._infer_prominent_text_from_context(context_text)
        if inferred_prominent_text:
            normalized = self.normalizer.normalize(
                context_text,
                source_name=source_name,
                prominent_text=inferred_prominent_text,
            )
        base_name = normalized.nombre_producto
        if not base_name:
            return []

        names = self._context_names(base_name, normalized, context_text)
        display_prominent_text = self._display_prominent_text(inferred_prominent_text)
        _append_unique(names, display_prominent_text)
        if (
            display_prominent_text
            and normalized.contenido_neto
            and _fold(normalized.contenido_neto) not in _fold(display_prominent_text)
        ):
            _append_unique(names, f"{display_prominent_text} {normalized.contenido_neto}")
        if normalized.contenido_neto and _fold(normalized.contenido_neto) not in _fold(base_name):
            _append_unique(names, f"{base_name} {normalized.contenido_neto}")
        self._append_ocr_text_name_variants(names, normalized, context_text, display_prominent_text)

        items: list[ProductSuggestionItem] = []
        for name in names[:3]:
            items.append(
                ProductSuggestionItem(
                    nombre_producto=name,
                    marca=normalized.marca,
                    tipo_producto=normalized.tipo_producto,
                    categoria_sugerida=normalized.categoria_sugerida,
                    source="ocr",
                )
            )
        return items

    def _display_prominent_text(self, prominent_text: str | None) -> str | None:
        if not prominent_text:
            return None
        cleaned = self._clean_ocr_name_line(prominent_text)
        if not cleaned:
            return None
        return self.normalizer._title_product_phrase(cleaned)

    def _should_infer_prominent_for_normalized(self, normalized) -> bool:
        if normalized.marca:
            return False
        folded_name = _fold(normalized.nombre_producto or "")
        folded_type = _fold(normalized.tipo_producto or "")
        if not folded_name:
            return True
        generic_names = {
            "antiacido",
            "analgesico",
            "antibiotico",
            "crema",
            "jarabe",
            "tableta",
            "capsula",
            "suspension",
        }
        if folded_name in generic_names or (folded_type and folded_name == folded_type):
            return True
        if re.search(r"\b(magaldrato|simeticona|mometasona|furoato|betametasona|clotrimazol)\b", folded_name):
            return True
        return False

    def _infer_prominent_text_from_context(self, context_text: str) -> str | None:
        candidates: list[tuple[int, int, str]] = []
        for index, raw_line in enumerate(context_text.splitlines()):
            line = self._clean_ocr_name_line(raw_line)
            if not line:
                continue
            folded = _fold(line)
            if len(folded) < 3:
                continue
            if re.fullmatch(r"[\d\s.,/%+-]+", folded):
                continue
            if re.search(r"\b(via|oral|laboratorio|laboratorios|contenido|neto|registro|dosis|suspension)\b", folded):
                continue
            if re.search(r"\b\d+(?:[.,]\d+)?\s*(mg|mcg|g|kg|ml|l|%)\b", folded):
                continue

            tokens = re.findall(r"[a-z0-9]+", folded)
            if not tokens:
                continue
            score = sum(len(token) for token in tokens)
            raw_alpha = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", "", line)
            if raw_alpha and raw_alpha.upper() == raw_alpha:
                score += 30
            if len(tokens) >= 2:
                score += 12
            if re.search(r"\b(antiacido|analgesico|jarabe|crema|tableta|capsula|suspension)\b", folded):
                score -= 22
            if re.search(r"\b(magaldrato|simeticona|mometasona|furoato|betametasona|clotrimazol)\b", folded):
                score -= 18
            candidates.append((score, -index, line))

        if not candidates:
            return None
        best = max(candidates)
        return best[2] if best[0] > 0 else None

    def _clean_ocr_name_line(self, raw_line: str) -> str:
        line = " ".join(raw_line.split())
        line = re.sub(r"[®™©]+", "", line)
        line = re.sub(r"(?<=[A-Za-zÁÉÍÓÚÜÑáéíóúüñ])['\"`´](?=[A-Za-zÁÉÍÓÚÜÑáéíóúüñ])", " ", line)
        line = re.sub(r"\s{2,}", " ", line)
        return line.strip(" -.,:;")

    def _context_names(self, base_name: str, normalized, context_text: str) -> list[str]:
        names: list[str] = []
        _append_unique(names, base_name)

        product_type = normalized.tipo_producto
        brand = normalized.marca
        if product_type and brand:
            folded_context = _fold(context_text)
            phrase_options = [
                ("rosas y magnolias", "Rosas y Magnolias"),
                ("cuidadoy suavidad", "Cuidado y Suavidad"),
                ("cuidado y suavidad", "Cuidado y Suavidad"),
                ("cuidado total", "Cuidado Total"),
                ("suavidad y fragancia prolongada", "Suavidad y Fragancia Prolongada"),
                ("suavidad fragancia prolongada", "Suavidad y Fragancia Prolongada"),
                ("protege el color y las fibras", "Protege Color y Fibras"),
                ("protege color fibras", "Protege Color y Fibras"),
                ("elcolory lasfibras", "Protege Color y Fibras"),
                ("lavanda", "Lavanda"),
                ("lawanda", "Lavanda"),
            ]
            for needle, label in phrase_options:
                if needle in folded_context:
                    _append_unique(names, f"{product_type} {brand} {label}")
            if "cuidado" in folded_context and "total" in folded_context:
                _append_unique(names, f"{product_type} {brand} Cuidado Total")
            if "protege" in folded_context and "color" in folded_context and "fibras" in folded_context:
                _append_unique(names, f"{product_type} {brand} Protege Color y Fibras")
            if "suavidad" in folded_context and "fragancia" in folded_context and "prolongada" in folded_context:
                _append_unique(names, f"{product_type} {brand} Suavidad y Fragancia Prolongada")

            if normalized.contenido_neto and _fold(normalized.contenido_neto) not in _fold(base_name):
                _append_unique(names, f"{base_name} {normalized.contenido_neto}")

            if normalized.presentacion:
                _append_unique(names, f"{product_type} {brand} {normalized.presentacion}")

            _append_unique(names, f"{product_type} {brand}")

        return names

    def _append_ocr_text_name_variants(
        self,
        names: list[str],
        normalized,
        context_text: str,
        prominent_text: str | None,
    ) -> None:
        useful_lines = self._useful_ocr_lines(context_text)
        prominent = " ".join((prominent_text or "").split())
        base = prominent or (names[0] if names else "")
        content = normalized.contenido_neto
        product_type = normalized.tipo_producto
        presentation = normalized.presentacion

        if base:
            if content:
                _append_unique(names, f"{base} {content}")
            if product_type and _fold(product_type) not in _fold(base):
                _append_unique(names, f"{base} {product_type}")
            if presentation and _fold(presentation) not in _fold(base):
                _append_unique(names, f"{base} {presentation}")

        for line in useful_lines:
            if len(names) >= 3:
                break
            if base and _fold(line) == _fold(base):
                continue
            if content and _fold(line) == _fold(content):
                continue
            if base:
                _append_unique(names, f"{base} {line}")
            else:
                _append_unique(names, line)

        if len(names) < 3 and product_type and content:
            _append_unique(names, f"{product_type} {content}")

    def _useful_ocr_lines(self, context_text: str) -> list[str]:
        lines: list[str] = []
        for raw_line in context_text.splitlines():
            line = " ".join(raw_line.split()).strip(" -.,:;")
            if not line:
                continue
            folded = _fold(line)
            if len(folded) < 3:
                continue
            if re.fullmatch(r"[\d\s.,/%+-]+", folded):
                continue
            if re.search(r"\b(via|laboratorio|laboratorios|contenido|neto|registro|dosis)\b", folded):
                continue
            if re.search(r"\b\d+(?:[.,]\d+)?\s*(mg|mcg|g|kg|ml|l|%)\b", folded):
                continue
            _append_unique(lines, line)
            if len(lines) >= 4:
                break
        return lines

    def _catalog_candidates(self, query: str) -> list[ProductSuggestionItem]:
        folded_query = _fold(query)
        items: list[tuple[int, ProductSuggestionItem]] = []

        for product_type, aliases in PRODUCT_TYPE_ALIASES.items():
            for alias in (product_type, *aliases):
                folded_alias = _fold(alias)
                if folded_query in folded_alias:
                    score = 0 if folded_alias.startswith(folded_query) else 1
                    items.append(
                        (
                            score,
                            ProductSuggestionItem(
                                nombre_producto=product_type,
                                tipo_producto=product_type,
                                categoria_sugerida=PRODUCT_TYPE_CATEGORIES.get(product_type),
                                source="catalog",
                            ),
                        )
                    )
                    break

        for brand in BRAND_ALIASES:
            for alias in (brand.canonical, *brand.aliases):
                folded_alias = _fold(alias)
                if folded_query in folded_alias:
                    score = 0 if folded_alias.startswith(folded_query) else 1
                    items.append(
                        (
                            score,
                            ProductSuggestionItem(
                                nombre_producto=brand.canonical,
                                marca=brand.canonical,
                                tipo_producto=BRAND_DEFAULT_PRODUCT_TYPES.get(brand.canonical),
                                categoria_sugerida=brand.category_hint,
                                source="catalog",
                            ),
                        )
                    )
                    break

        items.sort(key=lambda pair: (pair[0], len(pair[1].nombre_producto)))
        return [item for _, item in items]


def build_suggestion_service(repository: ProductRepository) -> ProductSuggestionService:
    return ProductSuggestionService(repository)
