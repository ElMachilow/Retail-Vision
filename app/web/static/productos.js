const apiBaseUrl = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

const tableBody = document.querySelector("#products-tbody");
const table = document.querySelector("#products-table");
const emptyMessage = document.querySelector("#products-empty");
const listStatus = document.querySelector("#list-status");
const backdrop = document.querySelector("#edit-backdrop");
const editForm = document.querySelector("#edit-form");
const cancelButton = document.querySelector("#cancel-button");
const saveButton = document.querySelector("#save-edit-button");
const editStatus = document.querySelector("#edit-status");

const editFields = {
  id: document.querySelector("#edit-id"),
  nombre_producto: document.querySelector("#edit-name"),
  tipo_producto: document.querySelector("#edit-type"),
  marca: document.querySelector("#edit-brand"),
  presentacion: document.querySelector("#edit-presentation"),
  contenido_neto: document.querySelector("#edit-content"),
  unidad_medida: document.querySelector("#edit-unit"),
  categoria_sugerida: document.querySelector("#edit-category"),
  codigo_barras: document.querySelector("#edit-barcode"),
  precio_venta: document.querySelector("#edit-price"),
};

let originalSnapshot = null;
let currentId = null;
let categorizationAbortController = null;

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value).replace(/[&<>"']/g, (char) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char];
  });
}

function formatPrice(value) {
  if (value === null || value === undefined) return "-";
  return `S/ ${Number(value).toFixed(2)}`;
}

async function loadProducts() {
  listStatus.textContent = "Cargando productos...";
  listStatus.classList.remove("is-error");
  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/productos`);
    if (!response.ok) throw new Error("No se pudo cargar la lista de productos.");
    const data = await response.json();
    renderTable(data.items || []);
    listStatus.textContent = `${(data.items || []).length} producto(s) registrados.`;
  } catch (error) {
    listStatus.classList.add("is-error");
    listStatus.textContent = error.message;
    table.hidden = true;
    emptyMessage.hidden = true;
  }
}

function renderTable(items) {
  if (!items.length) {
    table.hidden = true;
    emptyMessage.hidden = false;
    tableBody.innerHTML = "";
    return;
  }
  emptyMessage.hidden = true;
  table.hidden = false;
  tableBody.innerHTML = items
    .map(
      (item) => `<tr data-id="${item.id}">
        <td>${escapeHtml(item.nombre_producto)}</td>
        <td>${escapeHtml(item.marca || "-")}</td>
        <td>${escapeHtml(item.categoria_sugerida || "-")}</td>
        <td>${escapeHtml(item.contenido_neto || "-")}</td>
        <td>${escapeHtml(item.codigo_barras || "-")}</td>
        <td>${formatPrice(item.precio_venta)}</td>
      </tr>`,
    )
    .join("");
  tableBody.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("click", () => openEditor(Number(row.dataset.id)));
  });
}

function clearFieldErrors() {
  document.querySelectorAll(".field-error").forEach((node) => {
    node.hidden = true;
    node.textContent = "";
  });
  Object.values(editFields).forEach((input) => {
    if (input && input.classList) input.classList.remove("is-invalid");
  });
}

function setFieldError(name, message) {
  const node = document.querySelector(`.field-error[data-error-for="${name}"]`);
  if (node) {
    node.textContent = message;
    node.hidden = false;
  }
  if (editFields[name]) editFields[name].classList.add("is-invalid");
}

function fillEditor(record) {
  editFields.id.value = record.id;
  editFields.nombre_producto.value = record.nombre_producto || "";
  editFields.tipo_producto.value = record.tipo_producto || "";
  editFields.marca.value = record.marca || "";
  editFields.presentacion.value = record.presentacion || "";
  editFields.contenido_neto.value = record.contenido_neto || "";
  editFields.unidad_medida.value = record.unidad_medida || "";
  editFields.categoria_sugerida.value = record.categoria_sugerida || "";
  editFields.codigo_barras.value = record.codigo_barras || "";
  editFields.precio_venta.value =
    record.precio_venta === null || record.precio_venta === undefined
      ? ""
      : Number(record.precio_venta).toFixed(2);
}

async function openEditor(id) {
  clearFieldErrors();
  editStatus.textContent = "Cargando...";
  editStatus.classList.remove("is-error");
  backdrop.hidden = false;
  saveButton.disabled = true;
  currentId = id;
  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/productos/${id}`);
    if (!response.ok) throw new Error("No se pudo cargar el producto.");
    const record = await response.json();
    fillEditor(record);
    originalSnapshot = readPayload();
    editStatus.textContent = "";
  } catch (error) {
    editStatus.classList.add("is-error");
    editStatus.textContent = error.message;
  } finally {
    saveButton.disabled = false;
  }
}

function readPayload() {
  return {
    nombre_producto: editFields.nombre_producto.value.trim(),
    marca: editFields.marca.value.trim() || null,
    tipo_producto: editFields.tipo_producto.value.trim() || null,
    presentacion: editFields.presentacion.value.trim() || null,
    contenido_neto: editFields.contenido_neto.value.trim() || null,
    unidad_medida: editFields.unidad_medida.value.trim() || null,
    categoria_sugerida: editFields.categoria_sugerida.value.trim(),
    codigo_barras: editFields.codigo_barras.value.trim() || null,
    precio_venta:
      editFields.precio_venta.value === "" ? null : Number(editFields.precio_venta.value),
  };
}

function applyCategorization(item) {
  if (!item) return;
  if (item.marca && !editFields.marca.value) editFields.marca.value = item.marca;
  if (item.tipo_producto && !editFields.tipo_producto.value)
    editFields.tipo_producto.value = item.tipo_producto;
  if (item.presentacion && !editFields.presentacion.value)
    editFields.presentacion.value = item.presentacion;
  if (item.contenido_neto && !editFields.contenido_neto.value)
    editFields.contenido_neto.value = item.contenido_neto;
  if (item.unidad_medida && !editFields.unidad_medida.value)
    editFields.unidad_medida.value = item.unidad_medida;
  if (item.categoria_sugerida && !editFields.categoria_sugerida.value)
    editFields.categoria_sugerida.value = item.categoria_sugerida;
}

async function categorizeCurrentName() {
  const name = editFields.nombre_producto.value.trim();
  if (name.length < 3 || editFields.categoria_sugerida.value.trim()) return;
  if (categorizationAbortController) categorizationAbortController.abort();
  categorizationAbortController = new AbortController();
  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/productos/categorize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nombre_producto: name }),
      signal: categorizationAbortController.signal,
    });
    if (!response.ok) return;
    applyCategorization(await response.json());
  } catch (error) {
    if (error.name !== "AbortError") console.warn("No se pudo categorizar", error);
  }
}

function validatePayload(payload) {
  clearFieldErrors();
  let valid = true;
  if (!payload.nombre_producto) {
    setFieldError("nombre_producto", "Campo obligatorio.");
    valid = false;
  }
  if (!payload.categoria_sugerida) {
    setFieldError("categoria_sugerida", "Campo obligatorio.");
    valid = false;
  }
  if (!payload.contenido_neto) {
    setFieldError("contenido_neto", "Campo obligatorio.");
    valid = false;
  }
  if (
    payload.precio_venta === null ||
    Number.isNaN(payload.precio_venta) ||
    payload.precio_venta < 0
  ) {
    setFieldError("precio_venta", "Ingresa un precio ≥ 0.");
    valid = false;
  } else {
    payload.precio_venta = Math.round(payload.precio_venta * 100) / 100;
  }
  return valid;
}

function payloadEquals(a, b) {
  if (!a || !b) return false;
  return Object.keys(a).every((key) => (a[key] ?? null) === (b[key] ?? null));
}

function closeEditor() {
  backdrop.hidden = true;
  currentId = null;
  originalSnapshot = null;
  editStatus.textContent = "";
  editStatus.classList.remove("is-error");
  clearFieldErrors();
}

cancelButton.addEventListener("click", () => {
  const current = readPayload();
  if (!payloadEquals(current, originalSnapshot)) {
    if (!window.confirm("Hay cambios sin guardar. ¿Descartarlos?")) return;
  }
  closeEditor();
});

backdrop.addEventListener("click", (event) => {
  if (event.target === backdrop) {
    cancelButton.click();
  }
});

editForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (currentId === null) return;
  await categorizeCurrentName();
  const payload = readPayload();
  if (!validatePayload(payload)) {
    editStatus.classList.add("is-error");
    editStatus.textContent = "Revisa los campos obligatorios.";
    return;
  }
  saveButton.disabled = true;
  editStatus.classList.remove("is-error");
  editStatus.textContent = "Guardando...";
  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/productos/${currentId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      if (data.error_code === "DUPLICATE_BARCODE") {
        setFieldError("codigo_barras", "Ya existe un producto con este código.");
      } else if (data.error_code === "PRODUCT_NOT_FOUND") {
        editStatus.textContent = "El producto ya no existe.";
      }
      throw new Error(data.message || "No se pudo actualizar el producto.");
    }
    editStatus.textContent = "Actualizado.";
    await loadProducts();
    setTimeout(closeEditor, 600);
  } catch (error) {
    editStatus.classList.add("is-error");
    editStatus.textContent = error.message;
  } finally {
    saveButton.disabled = false;
  }
});

editFields.nombre_producto.addEventListener("blur", () => {
  categorizeCurrentName();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !backdrop.hidden) {
    cancelButton.click();
  }
});

loadProducts();
