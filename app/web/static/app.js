const form = document.querySelector("#recognition-form");
const imageInput = document.querySelector("#image-input");
const cameraInput = document.querySelector("#camera-input");
const dropZone = document.querySelector("#drop-zone");
const fileName = document.querySelector("#file-name");
const previewImage = document.querySelector("#preview-image");
const previewVideo = document.querySelector("#preview-video");
const previewEmpty = document.querySelector("#preview-empty");
const submitButton = document.querySelector("#submit-button");
const statusText = document.querySelector("#status-text");
const warningsBox = document.querySelector("#warnings");
const productForm = document.querySelector("#product-form");
const saveButton = document.querySelector("#save-button");
const saveStatus = document.querySelector("#save-status");
const suggestionsList = document.querySelector("#suggestions-list");
const webcamToggle = document.querySelector("#webcam-toggle");
const webcamActions = document.querySelector("#webcam-actions");
const webcamCapture = document.querySelector("#webcam-capture");
const webcamClose = document.querySelector("#webcam-close");
const webcamDevice = document.querySelector("#webcam-device");
const apiBaseUrl = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

const editableFields = {
  nombre_producto: document.querySelector("#product-name-input"),
  tipo_producto: document.querySelector("#product-type-input"),
  marca: document.querySelector("#brand-input"),
  presentacion: document.querySelector("#presentation-input"),
  contenido_neto: document.querySelector("#content-input"),
  unidad_medida: document.querySelector("#unit-input"),
  categoria_sugerida: document.querySelector("#category-input"),
  codigo_barras: document.querySelector("#barcode-input"),
  precio_venta: document.querySelector("#price-input"),
};

const traceFields = {
  model: document.querySelector("#model-value"),
  confidence: document.querySelector("#confidence-value"),
  bbox: document.querySelector("#bbox-value"),
  ocrConfidence: document.querySelector("#ocr-confidence-value"),
  time: document.querySelector("#time-value"),
  ocrText: document.querySelector("#ocr-text"),
};

let selectedFile = null;
let suggestionAbortController = null;
let suggestionDebounce = null;
let categorizationAbortController = null;
let latestOcrText = "";
let latestSourceName = "";
let latestProminentText = "";

function percent(value) {
  if (value === null || value === undefined) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("is-error", isError);
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading || !selectedFile;
  submitButton.textContent = isLoading ? "Analizando..." : "Analizar producto";
}

function clearTrace() {
  traceFields.model.textContent = "-";
  traceFields.confidence.textContent = "-";
  traceFields.bbox.textContent = "-";
  traceFields.ocrConfidence.textContent = "-";
  traceFields.time.textContent = "-";
  traceFields.ocrText.textContent = "-";
  warningsBox.hidden = true;
  warningsBox.textContent = "";
  latestOcrText = "";
  latestSourceName = "";
  latestProminentText = "";
}

function clearFieldErrors() {
  document.querySelectorAll(".field-error").forEach((node) => {
    node.hidden = true;
    node.textContent = "";
  });
  document.querySelectorAll(".field input").forEach((input) => {
    input.classList.remove("is-invalid");
  });
}

function setFieldError(fieldName, message) {
  const errorNode = document.querySelector(`.field-error[data-error-for="${fieldName}"]`);
  const input = editableFields[fieldName];
  if (errorNode) {
    errorNode.textContent = message;
    errorNode.hidden = false;
  }
  if (input) {
    input.classList.add("is-invalid");
  }
}

function fillEditableFromRecognition(product) {
  editableFields.nombre_producto.value = product.nombre_producto || "";
  editableFields.tipo_producto.value = product.tipo_producto || "";
  editableFields.marca.value = product.marca || "";
  editableFields.presentacion.value = product.presentacion || "";
  editableFields.contenido_neto.value = product.contenido_neto || "";
  editableFields.unidad_medida.value = product.unidad_medida || "";
  editableFields.categoria_sugerida.value = product.categoria_sugerida || "";
}

function updatePreview(file) {
  selectedFile = file;
  fileName.textContent = file ? file.name : "JPG, PNG o WEBP";
  submitButton.disabled = !file;
  clearTrace();
  hideSuggestions();

  if (file && webcamStream) {
    stopWebcam();
  }

  if (!file) {
    previewImage.hidden = true;
    previewImage.removeAttribute("src");
    previewEmpty.hidden = false;
    setStatus("Esperando imagen");
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  previewImage.onload = () => URL.revokeObjectURL(objectUrl);
  previewImage.src = objectUrl;
  previewImage.hidden = false;
  previewEmpty.hidden = true;
  setStatus("Imagen lista");
}

function renderResult(data) {
  const product = data.producto || {};
  const detection = data.deteccion || {};
  const ocr = data.ocr || {};
  const bbox = detection.bbox;

  fillEditableFromRecognition(product);

  traceFields.model.textContent = detection.model || "-";
  traceFields.confidence.textContent = percent(detection.confidence);
  traceFields.bbox.textContent = bbox
    ? `${bbox.x_min}, ${bbox.y_min}, ${bbox.x_max}, ${bbox.y_max}`
    : "-";
  traceFields.ocrConfidence.textContent = percent(ocr.average_confidence);
  traceFields.time.textContent = data.processing_ms ? `${data.processing_ms} ms` : "-";
  traceFields.ocrText.textContent = ocr.text || "-";
  latestOcrText = ocr.text || "";
  latestProminentText = ocr.prominent_text || "";
  latestSourceName = selectedFile?.name || "";

  if (Array.isArray(data.warnings) && data.warnings.length > 0) {
    warningsBox.hidden = false;
    warningsBox.textContent = data.warnings.join(" ");
  } else {
    warningsBox.hidden = true;
    warningsBox.textContent = "";
  }
}

async function analyzeProduct(event) {
  event.preventDefault();
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append("image", selectedFile);

  setLoading(true);
  setStatus("Procesando imagen con YOLO y OCR");

  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/products/recognize`, {
      method: "POST",
      headers: { "X-Trace-ID": `ui-${Date.now()}` },
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.message || "No se pudo analizar la imagen");
    }

    renderResult(data);
    setStatus("Producto analizado. Edita los campos y guarda.");
  } catch (error) {
    const hint =
      window.location.protocol === "file:"
        ? " Revisa que el servidor esté activo en http://127.0.0.1:8000/."
        : "";
    setStatus(`${error.message}.${hint}`, true);
  } finally {
    setLoading(false);
  }
}

function hideSuggestions() {
  suggestionsList.hidden = true;
  suggestionsList.innerHTML = "";
}

function renderSuggestions(items) {
  if (!items.length) {
    suggestionsList.innerHTML = '<li class="suggestion suggestion--empty">Sin sugerencias</li>';
    suggestionsList.hidden = false;
    return;
  }
  suggestionsList.innerHTML = items
    .slice(0, 3)
    .map((item, index) => {
      const meta = [item.marca, item.tipo_producto, item.categoria_sugerida]
        .filter(Boolean)
        .join(" · ");
      const safeName = item.nombre_producto.replace(/</g, "&lt;");
      const safeMeta = meta.replace(/</g, "&lt;");
      return `<li class="suggestion" role="option" data-index="${index}">
        <span class="suggestion-name">${safeName}</span>
        <span class="suggestion-meta">${safeMeta}</span>
      </li>`;
    })
    .join("");
  suggestionsList.hidden = false;
  suggestionsList.querySelectorAll(".suggestion").forEach((node, index) => {
    node.addEventListener("mousedown", (event) => {
      event.preventDefault();
      applySuggestion(items[index]);
    });
  });
}

function applySuggestion(item) {
  const input = editableFields.nombre_producto;
  input.value = item.nombre_producto;
  input.focus();
  const length = input.value.length;
  input.setSelectionRange(length, length);
  if (item.marca && !editableFields.marca.value) editableFields.marca.value = item.marca;
  if (item.tipo_producto && !editableFields.tipo_producto.value)
    editableFields.tipo_producto.value = item.tipo_producto;
  if (item.categoria_sugerida && !editableFields.categoria_sugerida.value)
    editableFields.categoria_sugerida.value = item.categoria_sugerida;
  hideSuggestions();
}

function applyCategorization(item) {
  if (!item) return;
  if (item.marca && !editableFields.marca.value) editableFields.marca.value = item.marca;
  if (item.tipo_producto && !editableFields.tipo_producto.value)
    editableFields.tipo_producto.value = item.tipo_producto;
  if (item.presentacion && !editableFields.presentacion.value)
    editableFields.presentacion.value = item.presentacion;
  if (item.contenido_neto && !editableFields.contenido_neto.value)
    editableFields.contenido_neto.value = item.contenido_neto;
  if (item.unidad_medida && !editableFields.unidad_medida.value)
    editableFields.unidad_medida.value = item.unidad_medida;
  if (item.categoria_sugerida && !editableFields.categoria_sugerida.value)
    editableFields.categoria_sugerida.value = item.categoria_sugerida;
}

async function categorizeCurrentName() {
  const name = editableFields.nombre_producto.value.trim();
  if (name.length < 3 || editableFields.categoria_sugerida.value.trim()) return;
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

async function fetchSuggestions(query) {
  if (suggestionAbortController) suggestionAbortController.abort();
  suggestionAbortController = new AbortController();
  try {
    const params = new URLSearchParams({ q: query, limit: "3" });
    if (latestOcrText) params.set("context", latestOcrText.slice(0, 2000));
    if (latestSourceName) params.set("source_name", latestSourceName.slice(0, 200));
    if (latestProminentText) params.set("prominent_text", latestProminentText.slice(0, 200));
    const response = await fetch(`${apiBaseUrl}/api/v1/productos/suggestions?${params}`, {
      signal: suggestionAbortController.signal,
    });
    if (!response.ok) {
      hideSuggestions();
      return;
    }
    const data = await response.json();
    renderSuggestions(data.items || []);
  } catch (error) {
    if (error.name !== "AbortError") {
      hideSuggestions();
    }
  }
}

editableFields.nombre_producto.addEventListener("input", (event) => {
  const query = event.target.value.trim();
  clearTimeout(suggestionDebounce);
  if (query.length < 3) {
    hideSuggestions();
    return;
  }
  suggestionDebounce = setTimeout(() => fetchSuggestions(query), 300);
});

editableFields.nombre_producto.addEventListener("blur", () => {
  categorizeCurrentName();
  setTimeout(hideSuggestions, 120);
});

editableFields.nombre_producto.addEventListener("focus", (event) => {
  const query = event.target.value.trim();
  if (query.length >= 3) fetchSuggestions(query);
});

function readPayload() {
  return {
    nombre_producto: editableFields.nombre_producto.value.trim(),
    marca: editableFields.marca.value.trim() || null,
    tipo_producto: editableFields.tipo_producto.value.trim() || null,
    presentacion: editableFields.presentacion.value.trim() || null,
    contenido_neto: editableFields.contenido_neto.value.trim() || null,
    unidad_medida: editableFields.unidad_medida.value.trim() || null,
    categoria_sugerida: editableFields.categoria_sugerida.value.trim(),
    codigo_barras: editableFields.codigo_barras.value.trim() || null,
    precio_venta: editableFields.precio_venta.value === "" ? null : Number(editableFields.precio_venta.value),
  };
}

function validatePayload(payload) {
  clearFieldErrors();
  let valid = true;
  if (!payload.nombre_producto) {
    setFieldError("nombre_producto", "El nombre es obligatorio.");
    valid = false;
  }
  if (!payload.categoria_sugerida) {
    setFieldError("categoria_sugerida", "La categoría es obligatoria.");
    valid = false;
  }
  if (!payload.contenido_neto) {
    setFieldError("contenido_neto", "Indica el contenido o unidad.");
    valid = false;
  }
  if (payload.precio_venta === null || Number.isNaN(payload.precio_venta) || payload.precio_venta < 0) {
    setFieldError("precio_venta", "Ingresa un precio ≥ 0.");
    valid = false;
  } else {
    payload.precio_venta = Math.round(payload.precio_venta * 100) / 100;
  }
  return valid;
}

async function saveProduct(event) {
  event.preventDefault();
  await categorizeCurrentName();
  const payload = readPayload();
  if (!validatePayload(payload)) {
    saveStatus.textContent = "Revisa los campos obligatorios.";
    saveStatus.classList.add("is-error");
    return;
  }
  saveButton.disabled = true;
  saveStatus.classList.remove("is-error");
  saveStatus.textContent = "Guardando...";

  try {
    const response = await fetch(`${apiBaseUrl}/api/v1/productos`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      if (data.error_code === "DUPLICATE_BARCODE") {
        setFieldError("codigo_barras", "Ya existe un producto con este código.");
      }
      throw new Error(data.message || "No se pudo guardar el producto.");
    }
    saveStatus.textContent = `Producto guardado (id ${data.id}).`;
  } catch (error) {
    saveStatus.classList.add("is-error");
    saveStatus.textContent = error.message;
  } finally {
    saveButton.disabled = false;
  }
}

let webcamStream = null;

function isWebcamActive() {
  return Boolean(webcamStream);
}

async function listVideoInputs() {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === "videoinput");
}

function populateDeviceSelect(devices, currentId) {
  webcamDevice.innerHTML = devices
    .map((d, idx) => {
      const label = d.label || `Cámara ${idx + 1}`;
      const selected = d.deviceId === currentId ? " selected" : "";
      return `<option value="${d.deviceId}"${selected}>${label.replace(/</g, "&lt;")}</option>`;
    })
    .join("");
  webcamDevice.hidden = devices.length <= 1;
}

async function startWebcam(deviceId) {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Tu navegador no soporta acceso a la cámara web.", true);
    return;
  }
  if (!window.isSecureContext) {
    setStatus(
      "La cámara solo funciona en HTTPS o localhost. Abre http://localhost:8000/ o http://127.0.0.1:8000/.",
      true,
    );
    return;
  }
  await stopWebcam();
  setStatus("Activando cámara…");
  const attempts = deviceId
    ? [{ video: { deviceId: { exact: deviceId } }, audio: false }]
    : [
        { video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
        { video: true, audio: false },
      ];
  let lastError = null;
  for (const constraints of attempts) {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
      lastError = null;
      break;
    } catch (error) {
      lastError = error;
      console.warn("getUserMedia falló para", constraints, error);
    }
  }
  if (!webcamStream) {
    const reason = describeMediaError(lastError);
    setStatus(`No se pudo abrir la cámara: ${reason}`, true);
    return;
  }
  try {
    previewVideo.srcObject = webcamStream;
    previewVideo.hidden = false;
    previewImage.hidden = true;
    previewEmpty.hidden = true;
    webcamActions.hidden = false;
    webcamToggle.classList.add("is-active");
    await previewVideo.play().catch(() => {});

    const devices = await listVideoInputs();
    const activeId = webcamStream.getVideoTracks()[0]?.getSettings()?.deviceId;
    populateDeviceSelect(devices, activeId);
    setStatus("Cámara activa. Encuadra el producto y pulsa Capturar.");
  } catch (error) {
    console.error("Error mostrando el stream", error);
    setStatus(`No se pudo mostrar el video: ${error.message}`, true);
    await stopWebcam();
  }
}

function describeMediaError(error) {
  if (!error) return "sin detalles";
  switch (error.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "el navegador bloqueó el acceso. Revisa permisos del sitio y del sistema (macOS: Ajustes → Privacidad → Cámara).";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "no se detectó ninguna cámara conectada.";
    case "NotReadableError":
    case "TrackStartError":
      return "la cámara está en uso por otra aplicación. Cierra Zoom/Teams/FaceTime e intenta de nuevo.";
    case "OverconstrainedError":
      return "la cámara seleccionada no soporta esa resolución.";
    case "SecurityError":
      return "el contexto no es seguro (usa localhost o HTTPS).";
    default:
      return `${error.name || "Error"}: ${error.message || "desconocido"}`;
  }
}

async function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
  }
  if (previewVideo.srcObject) {
    previewVideo.srcObject = null;
  }
  previewVideo.hidden = true;
  webcamActions.hidden = true;
  webcamToggle.classList.remove("is-active");
  if (!selectedFile) {
    previewEmpty.hidden = false;
  }
}

async function captureWebcamFrame() {
  if (!isWebcamActive()) return;
  const width = previewVideo.videoWidth;
  const height = previewVideo.videoHeight;
  if (!width || !height) {
    setStatus("La cámara aún no está lista. Espera un instante.", true);
    return;
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(previewVideo, 0, 0, width, height);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
  if (!blob) {
    setStatus("No se pudo capturar la imagen.", true);
    return;
  }
  const file = new File([blob], `webcam-${Date.now()}.jpg`, { type: "image/jpeg" });
  await stopWebcam();
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  updatePreview(file);
}

if (webcamToggle) {
  webcamToggle.addEventListener("click", () => {
    console.log("[webcam] click", { active: isWebcamActive(), secureContext: window.isSecureContext });
    if (isWebcamActive()) {
      stopWebcam();
      setStatus("Cámara cerrada.");
    } else {
      startWebcam();
    }
  });
} else {
  console.error("[webcam] #webcam-toggle no encontrado — recarga con Cmd+Shift+R para forzar HTML/JS nuevos.");
}

webcamCapture.addEventListener("click", () => {
  captureWebcamFrame();
});

webcamClose.addEventListener("click", () => {
  stopWebcam();
  setStatus("Cámara cerrada.");
});

webcamDevice.addEventListener("change", (event) => {
  startWebcam(event.target.value);
});

window.addEventListener("beforeunload", () => {
  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
  }
});

imageInput.addEventListener("change", () => {
  updatePreview(imageInput.files[0] || null);
});

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

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  const file = Array.from(event.dataTransfer.files).find((item) =>
    item.type.startsWith("image/"),
  );
  if (!file) {
    setStatus("Selecciona una imagen válida", true);
    return;
  }
  imageInput.files = event.dataTransfer.files;
  updatePreview(file);
});

form.addEventListener("submit", analyzeProduct);
productForm.addEventListener("submit", saveProduct);

clearTrace();
