const apiBaseUrl = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

const tableBody = document.querySelector("#review-table-body");
const emptyState = document.querySelector("#review-empty");
const listCount = document.querySelector("#list-count");
const detailEmpty = document.querySelector("#detail-empty");
const detailContent = document.querySelector("#detail-content");
const searchInput = document.querySelector("#admin-search-input");
const statusFilter = document.querySelector("#status-filter");
const categoryFilter = document.querySelector("#category-filter");
const confidenceFilter = document.querySelector("#confidence-filter");
const confidenceLabel = document.querySelector("#confidence-label");
const applyFilters = document.querySelector("#apply-filters");
const reviewStatus = document.querySelector("#review-status");
const selectAllRecords = document.querySelector("#select-all-records");
const selectedCount = document.querySelector("#selected-count");
const bulkDeleteButton = document.querySelector("#bulk-delete");

const metrics = {
  pending: document.querySelector("#metric-pending"),
  validated: document.querySelector("#metric-validated"),
  corrected: document.querySelector("#metric-corrected"),
  rejected: document.querySelector("#metric-rejected"),
  training: document.querySelector("#metric-training"),
  precision: document.querySelector("#metric-precision"),
};

const detail = {
  original: document.querySelector("#detail-original-image"),
  crop: document.querySelector("#detail-crop-image"),
  classBadge: document.querySelector("#detail-class-badge"),
  ocrConfidence: document.querySelector("#detail-ocr-confidence"),
  ocrText: document.querySelector("#detail-ocr-text"),
  suggestion: document.querySelector("#detail-ai-suggestion"),
};

const fields = {
  final_nombre_producto: document.querySelector("#review-name"),
  final_marca: document.querySelector("#review-brand"),
  final_tipo_producto: document.querySelector("#review-type"),
  final_presentacion: document.querySelector("#review-presentation"),
  final_contenido_neto: document.querySelector("#review-content"),
  final_unidad_medida: document.querySelector("#review-unit"),
  final_categoria_sugerida: document.querySelector("#review-category"),
  final_codigo_barras: document.querySelector("#review-barcode"),
  failure_reason: document.querySelector("#review-reason"),
  review_notes: document.querySelector("#review-notes"),
  use_for_training: document.querySelector("#review-training"),
};

let records = [];
let selectedId = null;
let searchDebounce = null;
let selectedRecordIds = new Set();
let categorizationAbortController = null;

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value).replace(/[&<>"']/g, (char) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char];
  });
}

function percent(value) {
  if (value === null || value === undefined) return "--";
  return `${Math.round(Number(value) * 100)}%`;
}

function statusLabel(status) {
  return {
    pending_review: "Pendiente",
    validated: "Validado",
    corrected: "Corregido",
    rejected: "Rechazado",
    duplicate: "Duplicado",
    ignored: "Ignorado",
    training_candidate: "Candidato",
    used_for_training: "Entrenado",
  }[status] || status;
}

function productName(item) {
  return item.final_nombre_producto || item.predicted_nombre_producto || "Producto sin nombre";
}

function brandName(item) {
  return item.final_marca || item.predicted_marca || "Sin marca";
}

function categoryName(item) {
  return item.final_categoria_sugerida || item.predicted_categoria_sugerida || "";
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  redirectIfUnauthorized(response);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.message || "No se pudo completar la operación.");
  return data;
}

function redirectIfUnauthorized(response) {
  if (response.status === 401) {
    window.location.href = "/login";
  }
}

async function loadStats() {
  const data = await fetchJson(`${apiBaseUrl}/api/v1/admin/reconocimientos/stats`);
  metrics.pending.textContent = data.pending_review;
  metrics.validated.textContent = data.validated;
  metrics.corrected.textContent = data.corrected;
  metrics.rejected.textContent = data.rejected;
  metrics.training.textContent = data.training_candidates;
  metrics.precision.textContent = `${Number(data.precision).toFixed(1)}%`;
}

function buildListUrl() {
  const params = new URLSearchParams({ limit: "200" });
  const status = statusFilter.value;
  const category = categoryFilter.value;
  const minConfidence = Number(confidenceFilter.value) / 100;
  const query = searchInput.value.trim();
  if (status !== "all") params.set("status", status);
  if (category !== "all") params.set("category", category);
  if (minConfidence > 0) params.set("min_confidence", String(minConfidence));
  if (query) params.set("q", query);
  return `${apiBaseUrl}/api/v1/admin/reconocimientos?${params}`;
}

async function loadRecords() {
  tableBody.innerHTML = `<tr><td colspan="7" class="table-loading">Cargando reconocimientos...</td></tr>`;
  try {
    const data = await fetchJson(buildListUrl());
    records = data.items || [];
    selectedRecordIds = new Set([...selectedRecordIds].filter((id) => records.some((item) => item.id === id)));
    renderCategoryOptions(records);
    renderTable(records);
    if (records.length && !selectedId) selectRecord(records[0].id);
    if (selectedId && !records.some((item) => item.id === selectedId)) {
      selectedId = null;
      if (records.length) selectRecord(records[0].id);
      else clearDetail();
    }
  } catch (error) {
    tableBody.innerHTML = `<tr><td colspan="7" class="table-loading is-error">${escapeHtml(error.message)}</td></tr>`;
  }
}

function renderCategoryOptions(items) {
  const current = categoryFilter.value;
  const categories = Array.from(new Set(items.map(categoryName).filter(Boolean))).sort();
  categoryFilter.innerHTML =
    `<option value="all">Todas las categorías</option>` +
    categories
      .map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`)
      .join("");
  if (categories.includes(current)) categoryFilter.value = current;
}

function renderTable(items) {
  listCount.textContent = `${items.length} registros`;
  emptyState.hidden = items.length > 0;
  if (!items.length) {
    tableBody.innerHTML = "";
    updateBulkControls();
    return;
  }
  tableBody.innerHTML = items
    .map((item) => {
      const selected = item.id === selectedId ? " is-selected" : "";
      return `<tr class="${selected}" data-id="${item.id}">
        <td>
          <input class="row-check" type="checkbox" data-select-id="${item.id}" aria-label="Seleccionar reconocimiento ${item.id}" ${selectedRecordIds.has(item.id) ? "checked" : ""}>
        </td>
        <td>#${item.id}</td>
        <td><img class="review-thumb" src="${item.image_url}" alt=""></td>
        <td>
          <strong>${escapeHtml(productName(item))}</strong>
          <span>${escapeHtml(categoryName(item) || "Sin categoría")} / ${escapeHtml(brandName(item))}</span>
        </td>
        <td>
          <b class="${(item.yolo_confidence || 0) < 0.5 ? "low-score" : "good-score"}">${percent(item.yolo_confidence)}</b>
          <b class="${(item.ocr_confidence || 0) < 0.5 ? "low-score" : "good-score"}">${percent(item.ocr_confidence)}</b>
        </td>
        <td><span class="status-pill status-${item.status}">${statusLabel(item.status)}</span></td>
        <td>
          <div class="table-actions">
            <button class="table-action" type="button">Revisar</button>
            <button class="table-action table-action--danger" type="button" data-delete-id="${item.id}">Eliminar</button>
          </div>
        </td>
      </tr>`;
    })
    .join("");
  tableBody.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("click", () => selectRecord(Number(row.dataset.id)));
  });
  tableBody.querySelectorAll("[data-select-id]").forEach((checkbox) => {
    checkbox.addEventListener("click", (event) => event.stopPropagation());
    checkbox.addEventListener("change", () => {
      toggleRecordSelection(Number(checkbox.dataset.selectId), checkbox.checked);
    });
  });
  tableBody.querySelectorAll("[data-delete-id]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteRecord(Number(button.dataset.deleteId));
    });
  });
  updateBulkControls();
}

function toggleRecordSelection(id, isSelected) {
  if (isSelected) selectedRecordIds.add(id);
  else selectedRecordIds.delete(id);
  updateBulkControls();
}

function updateBulkControls() {
  const visibleIds = records.map((item) => item.id);
  const selectedVisible = visibleIds.filter((id) => selectedRecordIds.has(id));
  selectedCount.textContent = `${selectedRecordIds.size} seleccionados`;
  bulkDeleteButton.disabled = selectedRecordIds.size === 0;
  if (!selectAllRecords) return;
  selectAllRecords.checked = visibleIds.length > 0 && selectedVisible.length === visibleIds.length;
  selectAllRecords.indeterminate = selectedVisible.length > 0 && selectedVisible.length < visibleIds.length;
}

function clearDetail() {
  detailContent.hidden = true;
  detailEmpty.hidden = false;
  selectedId = null;
}

function fillField(name, value) {
  const node = fields[name];
  if (!node) return;
  if (node.type === "checkbox") node.checked = Boolean(value);
  else node.value = value || "";
}

function currentRecord() {
  return records.find((record) => record.id === selectedId) || null;
}

function applyCategorization(item) {
  if (!item) return;
  if (item.marca && !fields.final_marca.value) fields.final_marca.value = item.marca;
  if (item.tipo_producto && !fields.final_tipo_producto.value) {
    fields.final_tipo_producto.value = item.tipo_producto;
  }
  if (item.presentacion && !fields.final_presentacion.value) {
    fields.final_presentacion.value = item.presentacion;
  }
  if (item.contenido_neto && !fields.final_contenido_neto.value) {
    fields.final_contenido_neto.value = item.contenido_neto;
  }
  if (item.unidad_medida && !fields.final_unidad_medida.value) {
    fields.final_unidad_medida.value = item.unidad_medida;
  }
  if (item.categoria_sugerida && !fields.final_categoria_sugerida.value) {
    fields.final_categoria_sugerida.value = item.categoria_sugerida;
  }

  const record = currentRecord();
  if (record && item.categoria_sugerida && !categoryName(record)) {
    record.predicted_categoria_sugerida = item.categoria_sugerida;
    record.predicted_marca = record.predicted_marca || item.marca;
    record.predicted_tipo_producto = record.predicted_tipo_producto || item.tipo_producto;
    renderCategoryOptions(records);
    renderTable(records);
  }
}

async function categorizeCurrentReview() {
  const name = fields.final_nombre_producto.value.trim();
  if (name.length < 3 || fields.final_categoria_sugerida.value.trim()) return;
  const record = currentRecord();
  if (categorizationAbortController) categorizationAbortController.abort();
  categorizationAbortController = new AbortController();
  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/productos/categorize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        nombre_producto: name,
        context: (record?.ocr_text || "").slice(0, 2000),
      }),
      signal: categorizationAbortController.signal,
    });
    if (!response.ok) return;
    applyCategorization(await response.json());
  } catch (error) {
    if (error.name !== "AbortError") console.warn("No se pudo categorizar", error);
  }
}

function selectRecord(id) {
  const item = records.find((record) => record.id === id);
  if (!item) return;
  selectedId = id;
  renderTable(records);
  detailEmpty.hidden = true;
  detailContent.hidden = false;

  detail.original.src = item.image_url;
  detail.crop.src = item.image_url;
  detail.classBadge.textContent = item.yolo_class_name || "label";
  detail.ocrConfidence.textContent = `Confianza: ${percent(item.ocr_confidence)}`;
  detail.ocrText.textContent = item.ocr_text || "Sin texto OCR.";
  detail.suggestion.textContent = productName(item);

  fillField("final_nombre_producto", item.final_nombre_producto || item.predicted_nombre_producto);
  fillField("final_marca", item.final_marca || item.predicted_marca);
  fillField("final_tipo_producto", item.final_tipo_producto || item.predicted_tipo_producto);
  fillField("final_presentacion", item.final_presentacion || item.predicted_presentacion);
  fillField("final_contenido_neto", item.final_contenido_neto || item.predicted_contenido_neto);
  fillField("final_unidad_medida", item.final_unidad_medida || item.predicted_unidad_medida);
  fillField("final_categoria_sugerida", item.final_categoria_sugerida || item.predicted_categoria_sugerida);
  fillField("final_codigo_barras", item.final_codigo_barras);
  fillField("failure_reason", item.failure_reason);
  fillField("review_notes", item.review_notes);
  fillField("use_for_training", item.use_for_training);
  categorizeCurrentReview();
  reviewStatus.textContent = "";
  reviewStatus.classList.remove("is-error");
}

function readReviewPayload(status) {
  return {
    status,
    final_nombre_producto: fields.final_nombre_producto.value.trim() || null,
    final_marca: fields.final_marca.value.trim() || null,
    final_tipo_producto: fields.final_tipo_producto.value.trim() || null,
    final_presentacion: fields.final_presentacion.value.trim() || null,
    final_contenido_neto: fields.final_contenido_neto.value.trim() || null,
    final_unidad_medida: fields.final_unidad_medida.value.trim() || null,
    final_categoria_sugerida: fields.final_categoria_sugerida.value.trim() || null,
    final_codigo_barras: fields.final_codigo_barras.value.trim() || null,
    failure_reason: fields.failure_reason.value || null,
    review_notes: fields.review_notes.value.trim() || null,
    use_for_training: fields.use_for_training.checked,
    linked_product_id: null,
  };
}

async function submitReview(status) {
  if (!selectedId) return;
  reviewStatus.textContent = "Guardando revisión...";
  reviewStatus.classList.remove("is-error");
  try {
    await categorizeCurrentReview();
    const data = await fetchJson(`${apiBaseUrl}/api/v1/admin/reconocimientos/${selectedId}/review`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readReviewPayload(status)),
    });
    const index = records.findIndex((item) => item.id === selectedId);
    if (index >= 0) records[index] = data;
    reviewStatus.textContent = `Revisión guardada como ${statusLabel(data.status)}.`;
    await loadStats();
    renderTable(records);
    selectRecord(data.id);
  } catch (error) {
    reviewStatus.classList.add("is-error");
    reviewStatus.textContent = error.message;
  }
}

async function deleteRecord(id) {
  const item = records.find((record) => record.id === id);
  const name = item ? productName(item) : `#${id}`;
  const confirmed = window.confirm(`Eliminar reconocimiento ${name}? Esta accion no se puede deshacer.`);
  if (!confirmed) return;

  reviewStatus.classList.remove("is-error");
  reviewStatus.textContent = "Eliminando reconocimiento...";
  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/admin/reconocimientos/${id}`, {
      method: "DELETE",
    });
    redirectIfUnauthorized(response);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.message || "No se pudo eliminar el reconocimiento.");
    }
    if (selectedId === id) clearDetail();
    selectedRecordIds.delete(id);
    records = records.filter((record) => record.id !== id);
    await loadStats();
    renderTable(records);
    if (!selectedId && records.length) selectRecord(records[0].id);
    reviewStatus.textContent = "Reconocimiento eliminado.";
  } catch (error) {
    reviewStatus.classList.add("is-error");
    reviewStatus.textContent = error.message;
  }
}

async function deleteSelectedRecords() {
  const ids = [...selectedRecordIds];
  if (!ids.length) return;
  const confirmed = window.confirm(`Eliminar ${ids.length} reconocimiento(s)? Esta accion no se puede deshacer.`);
  if (!confirmed) return;

  reviewStatus.classList.remove("is-error");
  reviewStatus.textContent = "Eliminando reconocimientos seleccionados...";
  bulkDeleteButton.disabled = true;
  try {
    const failed = [];
    for (const id of ids) {
      const response = await fetch(`${apiBaseUrl}/api/v1/admin/reconocimientos/${id}`, {
        method: "DELETE",
      });
      redirectIfUnauthorized(response);
      if (!response.ok) failed.push(id);
    }
    selectedRecordIds.clear();
    if (selectedId && ids.includes(selectedId)) clearDetail();
    records = records.filter((record) => !ids.includes(record.id));
    await loadStats();
    renderTable(records);
    if (!selectedId && records.length) selectRecord(records[0].id);
    reviewStatus.textContent = failed.length
      ? `No se pudieron eliminar ${failed.length} reconocimiento(s).`
      : "Reconocimientos seleccionados eliminados.";
    reviewStatus.classList.toggle("is-error", failed.length > 0);
  } catch (error) {
    reviewStatus.classList.add("is-error");
    reviewStatus.textContent = error.message;
  } finally {
    updateBulkControls();
  }
}

document.querySelectorAll("[data-action]").forEach((button) => {
  button.addEventListener("click", () => submitReview(button.dataset.action));
});

fields.final_nombre_producto.addEventListener("blur", () => {
  categorizeCurrentReview();
});

selectAllRecords.addEventListener("change", () => {
  if (selectAllRecords.checked) {
    records.forEach((item) => selectedRecordIds.add(item.id));
  } else {
    records.forEach((item) => selectedRecordIds.delete(item.id));
  }
  renderTable(records);
});

bulkDeleteButton.addEventListener("click", deleteSelectedRecords);

confidenceFilter.addEventListener("input", () => {
  confidenceLabel.textContent = `${confidenceFilter.value}%`;
});

applyFilters.addEventListener("click", loadRecords);
statusFilter.addEventListener("change", loadRecords);
categoryFilter.addEventListener("change", loadRecords);

searchInput.addEventListener("input", () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(loadRecords, 250);
});

async function boot() {
  await loadStats().catch(() => {});
  await loadRecords();
}

boot();
