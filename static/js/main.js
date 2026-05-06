const form = document.getElementById("predictForm");
const imageInput = document.getElementById("imageInput");
const previewWrapper = document.getElementById("previewWrapper");
const previewImage = document.getElementById("previewImage");

const loadingState = document.getElementById("loadingState");
const resultState = document.getElementById("resultState");
const errorState = document.getElementById("errorState");
const emptyState = document.getElementById("emptyState");

const resultLabel = document.getElementById("resultLabel");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceNotice = document.getElementById("confidenceNotice");
const topPredictions = document.getElementById("topPredictions");
const predictButton = document.getElementById("predictButton");

// Reset cards to default UI state.
function resetStates() {
  loadingState.classList.add("d-none");
  resultState.classList.add("d-none");
  errorState.classList.add("d-none");
  emptyState.classList.remove("d-none");
}

function showLoading() {
  loadingState.classList.remove("d-none");
  resultState.classList.add("d-none");
  errorState.classList.add("d-none");
  emptyState.classList.add("d-none");
}

function showResult(label, confidence) {
  loadingState.classList.add("d-none");
  errorState.classList.add("d-none");
  resultState.classList.remove("d-none");
  emptyState.classList.add("d-none");

  const percentage = (confidence * 100).toFixed(2);
  resultLabel.textContent = label;
  confidenceText.textContent = `${percentage}%`;
  confidenceBar.style.width = `${percentage}%`;
}

function renderTopPredictions(predictions) {
  topPredictions.innerHTML = "";

  predictions.forEach((prediction, index) => {
    const row = document.createElement("div");
    row.className = "top-prediction-row";
    row.innerHTML = `
      <div class="d-flex justify-content-between align-items-center mb-1">
        <span class="fw-semibold">${index + 1}. ${prediction.label}</span>
        <span class="text-muted">${(prediction.confidence * 100).toFixed(2)}%</span>
      </div>
      <div class="progress top-prediction-bar" role="progressbar" aria-label="Top prediction confidence">
        <div class="progress-bar bg-success" style="width: ${(prediction.confidence * 100).toFixed(2)}%"></div>
      </div>
    `;
    topPredictions.appendChild(row);
  });
}

function showConfidenceNotice(isLowConfidence) {
  if (!isLowConfidence) {
    confidenceNotice.classList.add("d-none");
    confidenceNotice.textContent = "";
    return;
  }

  confidenceNotice.textContent = "Low confidence prediction. The image may be unclear or the item may be out of training scope.";
  confidenceNotice.classList.remove("d-none");
}

function showError(message) {
  loadingState.classList.add("d-none");
  resultState.classList.add("d-none");
  emptyState.classList.add("d-none");
  errorState.classList.remove("d-none");
  errorState.textContent = message;
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) {
    previewWrapper.classList.add("d-none");
    previewImage.removeAttribute("src");
    return;
  }

  const reader = new FileReader();
  // Show client-side image preview before upload.
  reader.onload = (event) => {
    previewImage.src = event.target.result;
    previewWrapper.classList.remove("d-none");
  };
  reader.readAsDataURL(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files[0];
  if (!file) {
    showError("Please choose an image before predicting.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  showLoading();
  predictButton.disabled = true;

  try {
    // Send image to Flask backend and render JSON response.
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      showError(data.error || "Prediction failed. Please try again.");
      return;
    }

    showResult(data.label, data.confidence);
    showConfidenceNotice(Boolean(data.is_low_confidence));
    renderTopPredictions(data.top_predictions || []);
  } catch (error) {
    showError("Network error. Please check the server and try again.");
  } finally {
    predictButton.disabled = false;
  }
});

resetStates();
