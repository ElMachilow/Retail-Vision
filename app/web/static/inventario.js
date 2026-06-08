const apiBaseUrl = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

const sessionForm = document.querySelector("#session-form");
const sessionNameInput = document.querySelector("#session-name");
const sessionSelect = document.querySelector("#session-select");
const sessionStatus = document.querySelector("#session-status");
const closeSessionButton = document.querySelector("#close-session-button");

const recognitionForm = document.querySelector("#inventory-recognition-form");
const imageInput = document.querySelector("#inventory-image-input");
const cameraInput = document.querySelector("#inventory-camera-input");
const dropZone = document.querySelector("#inventory-drop-zone");
const fileName = document.querySelector("#inventory-file-name");
const previewImage = document.querySelector("#inventory-preview-image");
const previewEmpty = document.querySelector("#inventory-preview-empty");
const analyzeButton = document.querySelector("#inventory-analyze-button");
const statusText = document.querySelector("#inventory-status");
const warningsBox = document.querySelector("#inventory-warnings");

const itemForm = document.querySelector("#inventory-item-form");
const fields = {
  recognition_event_id: document.querySelector("#recognition-event-id"),
  product_id: document.querySelector("#matching-product-id"),
  nombre_producto: document.querySelector("#inventory-name"),
  marca: document.querySelector("#inventory-brand"),
  tipo_producto: document.querySelector("#inventory-type"),
  categoria: document.querySelector("#inventory-category"),
  contenido_neto: document.querySelector("#inventory-content"),
  unidad_medida: document.querySelector("#inventory-unit"),
  ubicacion: document.querySelector("#inventory-location"),
  cantidad: document.querySelector("#inventory-quantity"),
};
const saveStatus = document.querySelector("#inventory-save-status");
const summaryBox = document.querySelector("#inventory-summary");
const totalLabel = document.querySelector("#inventory-total-label");
const countLabel = document.querySelector("#inventory-count-label");
const itemsTable = document.querySelector("#inventory-items-table");
const itemsBody = document.querySelector("#inventory-items-body");
const emptyState = document.querySelector("#inventory-empty");

let selectedFile = null;
let sessions = [];

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value).replace(/[&<>"']/g, (char) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char];
  });
}

function activeSessionId() {
  return Number(sessionSelect.value || 0);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.message || "No se pudo completar la operacion.");
  return data;
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("is-error", isError);
}

function setSessionStatus(message, isError = false) {
  sessionStatus.textContent = message;
  sessionStatus.classList.toggle("is-error", isError);
}

function updateAnalyzeState() {
  analyzeButton.disabled = !selectedFile || !activeSessionId();
}

function todaySessionName() {
  const now = new Date();
  return `Inventario ${now.toLocaleDateString("es-PE")}`;
}

async function loadSessions() {
  const data = await fetchJson(`${apiBaseUrl}/api/v1/inventory/sessions`);
  sessions = data.items || [];
  if (!sessions.length) {
    const created = await fetchJson(`${apiBaseUrl}/api/v1/inventory/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nombre: todaySessionName() }),
    });
    sessions = [created];
  }
  renderSessions();
  await refreshInventory();
}

function renderSessions() {
  sessionSelect.innerHTML = sessions
    .map((session) => {
      const label = `${session.nombre} #${session.id}${session.estado === "closed" ? " (cerrada)" : ""}`;
      return `<option value="${session.id}">${escapeHtml(label)}</option>`;
    })
    .join("");
  updateAnalyzeState();
}

function updatePreview(file) {
  selectedFile = file;
  fileName.textContent = file ? file.name : "JPG, PNG o WEBP";
  updateAnalyzeState();
  if (!file) {
    previewImage.hidden = true;
    previewImage.removeAttribute("src");
    previewEmpty.hidden = false;
    setStatus("Esperando foto");
    return;
  }
  const objectUrl = URL.createObjectURL(file);
  previewImage.onload = () => URL.revokeObjectURL(objectUrl);
  previewImage.src = objectUrl;
  previewImage.hidden = false;
  previewEmpty.hidden = true;
  setStatus("Foto lista");
}

function fillProduct(product, data) {
  fields.recognition_event_id.value = data.recognition_event_id || "";
  fields.product_id.value = data.matching_product_id || "";
  fields.nombre_producto.value = product.nombre_producto || "";
  fields.marca.value = product.marca || "";
  fields.tipo_producto.value = product.tipo_producto || "";
  fields.categoria.value = product.categoria_sugerida || "";
  fields.contenido_neto.value = product.contenido_neto || "";
  fields.unidad_medida.value = product.unidad_medida || "";
  fields.cantidad.value = "1";
  if (Array.isArray(data.warnings) && data.warnings.length) {
    warningsBox.hidden = false;
    warningsBox.textContent = data.warnings.join(" ");
  } else {
    warningsBox.hidden = true;
    warningsBox.textContent = "";
  }
}

async function recognizePhoto(event) {
  event.preventDefault();
  if (!selectedFile || !activeSessionId()) return;

  const formData = new FormData();
  formData.append("image", selectedFile);
  analyzeButton.disabled = true;
  setStatus("Reconociendo producto...");
  try {
    const data = await fetchJson(
      `${apiBaseUrl}/api/v1/inventory/sessions/${activeSessionId()}/items/recognize`,
      {
        method: "POST",
        headers: { "X-Trace-ID": `inventory-${Date.now()}` },
        body: formData,
      },
    );
    fillProduct(data.producto || {}, data);
    setStatus("Producto reconocido. Confirma cantidad y guarda.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    updateAnalyzeState();
  }
}

function readItemPayload() {
  return {
    product_id: fields.product_id.value ? Number(fields.product_id.value) : null,
    recognition_event_id: fields.recognition_event_id.value
      ? Number(fields.recognition_event_id.value)
      : null,
    nombre_producto: fields.nombre_producto.value.trim(),
    marca: fields.marca.value.trim() || null,
    tipo_producto: fields.tipo_producto.value.trim() || null,
    categoria: fields.categoria.value.trim() || null,
    contenido_neto: fields.contenido_neto.value.trim() || null,
    unidad_medida: fields.unidad_medida.value.trim() || null,
    cantidad: Number(fields.cantidad.value),
    ubicacion: fields.ubicacion.value.trim() || null,
  };
}

async function saveItem(event) {
  event.preventDefault();
  const payload = readItemPayload();
  if (!activeSessionId()) {
    saveStatus.textContent = "Crea o selecciona una sesion.";
    saveStatus.classList.add("is-error");
    return;
  }
  if (!payload.nombre_producto || !Number.isInteger(payload.cantidad) || payload.cantidad < 1) {
    saveStatus.textContent = "Revisa producto y cantidad.";
    saveStatus.classList.add("is-error");
    return;
  }
  saveStatus.classList.remove("is-error");
  saveStatus.textContent = "Guardando conteo...";
  try {
    const item = await fetchJson(`${apiBaseUrl}/api/v1/inventory/sessions/${activeSessionId()}/items`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    saveStatus.textContent = `Conteo guardado: ${item.cantidad} unidad(es).`;
    await refreshInventory();
  } catch (error) {
    saveStatus.classList.add("is-error");
    saveStatus.textContent = error.message;
  }
}

async function refreshInventory() {
  const sessionId = activeSessionId();
  if (!sessionId) return;
  const [summary, items] = await Promise.all([
    fetchJson(`${apiBaseUrl}/api/v1/inventory/sessions/${sessionId}/summary`),
    fetchJson(`${apiBaseUrl}/api/v1/inventory/sessions/${sessionId}/items`),
  ]);
  renderSummary(summary);
  renderItems(items.items || []);
}

function renderSummary(summary) {
  totalLabel.textContent = `${summary.total_unidades} unidades`;
  if (!summary.categorias.length) {
    summaryBox.innerHTML = '<p class="products-empty">Sin conteos registrados.</p>';
    return;
  }
  summaryBox.innerHTML = summary.categorias
    .map(
      (item) => `<article class="inventory-summary-item">
        <strong>${escapeHtml(item.categoria)}</strong>
        <span>${item.productos} producto(s)</span>
        <b>${item.unidades}</b>
      </article>`,
    )
    .join("");
}

function renderItems(items) {
  countLabel.textContent = `${items.length} items`;
  emptyState.hidden = items.length > 0;
  itemsTable.hidden = !items.length;
  itemsBody.innerHTML = items
    .map(
      (item) => `<tr>
        <td>
          <strong>${escapeHtml(item.nombre_producto)}</strong>
          <span>${escapeHtml(item.marca || item.tipo_producto || "")}</span>
        </td>
        <td>${escapeHtml(item.categoria || "Sin categoria")}</td>
        <td>${item.cantidad}</td>
        <td>${escapeHtml(item.ubicacion || "-")}</td>
      </tr>`,
    )
    .join("");
}

sessionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const nombre = sessionNameInput.value.trim();
  if (!nombre) return;
  setSessionStatus("Creando sesion...");
  try {
    const created = await fetchJson(`${apiBaseUrl}/api/v1/inventory/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nombre }),
    });
    sessions.unshift(created);
    renderSessions();
    sessionSelect.value = String(created.id);
    sessionNameInput.value = "";
    setSessionStatus("Sesion creada.");
    await refreshInventory();
  } catch (error) {
    setSessionStatus(error.message, true);
  }
});

closeSessionButton.addEventListener("click", async () => {
  if (!activeSessionId()) return;
  setSessionStatus("Cerrando sesion...");
  try {
    const closed = await fetchJson(`${apiBaseUrl}/api/v1/inventory/sessions/${activeSessionId()}/close`, {
      method: "PUT",
    });
    sessions = sessions.map((item) => (item.id === closed.id ? closed : item));
    renderSessions();
    sessionSelect.value = String(closed.id);
    setSessionStatus("Sesion cerrada.");
  } catch (error) {
    setSessionStatus(error.message, true);
  }
});

sessionSelect.addEventListener("change", refreshInventory);

imageInput.addEventListener("change", () => updatePreview(imageInput.files[0] || null));
cameraInput.addEventListener("change", () => {
  const file = cameraInput.files[0] || null;
  if (file) {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    imageInput.files = transfer.files;
  }
  updatePreview(file);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("is-dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  const file = Array.from(event.dataTransfer.files).find((item) => item.type.startsWith("image/"));
  if (!file) {
    setStatus("Selecciona una imagen valida", true);
    return;
  }
  imageInput.files = event.dataTransfer.files;
  updatePreview(file);
});

recognitionForm.addEventListener("submit", recognizePhoto);
itemForm.addEventListener("submit", saveItem);

loadSessions().catch((error) => setSessionStatus(error.message, true));
