const API_URL = "http://127.0.0.1:8000";

const CLASS_META = {
  glioma:     { label: "Glioma",     badge: "Tumor detected", key: "glioma" },
  meningioma: { label: "Meningioma", badge: "Tumor detected", key: "meningioma" },
  pituitary:  { label: "Pituitary",  badge: "Tumor detected", key: "pituitary" },
  notumor:    { label: "No tumor",   badge: "Clear scan",     key: "notumor" },
};

const fileInput   = document.getElementById("fileInput");
const browseBtn   = document.getElementById("browseBtn");
const analyzeBtn  = document.getElementById("analyzeBtn");
const viewer      = document.getElementById("viewer");
const viewerEmpty = document.getElementById("viewerEmpty");
const previewImg  = document.getElementById("previewImage");
const scanLine    = document.getElementById("scanLine");
const fileNameEl  = document.getElementById("fileName");

const reportIdle   = document.getElementById("reportIdle");
const reportResult = document.getElementById("reportResult");
const verdictCard  = document.getElementById("verdictCard");
const verdictLabel = document.getElementById("verdictLabel");
const verdictBadge = document.getElementById("verdictBadge");
const confidenceValue = document.getElementById("confidenceValue");
const barsContainer = document.getElementById("bars");
const apiStatusEl   = document.getElementById("apiStatus");

let selectedFile = null;

viewer.addEventListener("click", () => fileInput.click());
browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
});

["dragenter", "dragover"].forEach(evt =>
  viewer.addEventListener(evt, (e) => { e.preventDefault(); viewer.style.borderColor = "var(--violet)"; })
);
["dragleave", "drop"].forEach(evt =>
  viewer.addEventListener(evt, (e) => { e.preventDefault(); viewer.style.borderColor = "var(--panel-border)"; })
);
viewer.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) return;
  selectedFile = file;

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.hidden = false;
  viewerEmpty.hidden = true;

  fileNameEl.textContent = file.name;
  analyzeBtn.disabled = false;
  resetReport();
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  analyzeBtn.querySelector("span").textContent = "Analyzing…";
  scanLine.classList.add("active");
  resetReport();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`Server responded ${res.status}`);
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    renderError(err);
  } finally {
    scanLine.classList.remove("active");
    analyzeBtn.disabled = false;
    analyzeBtn.querySelector("span").textContent = "Analyze scan";
  }
});

function resetReport() {
  reportIdle.hidden = false;
  reportResult.hidden = true;
  barsContainer.innerHTML = "";
}

function renderResult(data) {
  const meta = CLASS_META[data.prediction] || { label: data.prediction, badge: "Result", key: "glioma" };

  reportIdle.hidden = true;
  reportResult.hidden = false;

  verdictCard.className = "verdict c-" + meta.key;
  verdictLabel.textContent = meta.label;
  verdictBadge.textContent = meta.badge;
  verdictBadge.className = "verdict-badge c-" + meta.key;
  confidenceValue.textContent = `${data.confidence.toFixed(1)}%`;

  const probs = data.probabilities || { [data.prediction]: data.confidence };
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  barsContainer.innerHTML = "";
  sorted.forEach(([cls, value]) => {
    const clsMeta = CLASS_META[cls] || { label: cls, key: cls };

    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-row-top">
        <span>${clsMeta.label}</span>
        <span class="mono">${value.toFixed(1)}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill f-${clsMeta.key}" style="width:0%"></div>
      </div>
    `;
    barsContainer.appendChild(row);

    requestAnimationFrame(() => {
      row.querySelector(".bar-fill").style.width = `${value}%`;
    });
  });
}

function renderError(err) {
  reportIdle.hidden = false;
  reportResult.hidden = true;
  reportIdle.innerHTML = `<p style="color:var(--red)">Could not reach the model server.<br><span class="mono" style="font-size:11px;color:var(--text-muted)">${err.message}</span></p>`;
}

async function checkApiStatus() {
  try {
    const res = await fetch(`${API_URL}/`, { method: "GET" });
    apiStatusEl.textContent = res.ok ? "online" : "unreachable";
  } catch {
    apiStatusEl.textContent = "offline";
  }
}
checkApiStatus();
