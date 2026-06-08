from dataclasses import dataclass

from app.services.normalizer import ProductTextNormalizer


@dataclass(frozen=True)
class ProductCategorization:
    nombre_producto: str
    marca: str | None = None
    tipo_producto: str | None = None
    presentacion: str | None = None
    contenido_neto: str | None = None
    unidad_medida: str | None = None
    categoria_sugerida: str | None = None
    warnings: list[str] | None = None


class ProductCategorizer:
    def __init__(self) -> None:
        self.normalizer = ProductTextNormalizer()

    def categorize(
        self,
        nombre_producto: str,
        context_text: str | None = None,
    ) -> ProductCategorization:
        context = "\n".join(
            part
            for part in (
                nombre_producto,
                context_text,
            )
            if part
        )
        normalized = self.normalizer.normalize(
            context,
            prominent_text=nombre_producto,
        )
        return ProductCategorization(
            nombre_producto=nombre_producto.strip(),
            marca=normalized.marca,
            tipo_producto=normalized.tipo_producto,
            presentacion=normalized.presentacion,
            contenido_neto=normalized.contenido_neto,
            unidad_medida=normalized.unidad_medida,
            categoria_sugerida=normalized.categoria_sugerida,
            warnings=normalized.warnings,
        )

    def enrich_payload(self, payload: dict) -> dict:
        name = str(payload.get("nombre_producto") or "").strip()
        if not name:
            return payload

        categorized = self.categorize(name)
        enriched = dict(payload)
        for field in (
            "marca",
            "tipo_producto",
            "presentacion",
            "contenido_neto",
            "unidad_medida",
            "categoria_sugerida",
        ):
            if not enriched.get(field):
                enriched[field] = getattr(categorized, field)
        return enriched


def build_categorizer() -> ProductCategorizer:
    return ProductCategorizer()
