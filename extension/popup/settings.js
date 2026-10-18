const DEFAULT_BACKEND_URL = "https://api.deepshield.app";

const backendUrlInput = document.getElementById("backendUrl");
const testConnectionBtn = document.getElementById("testConnection");
const testResult = document.getElementById("testResult");
const currentUrlEl = document.getElementById("currentUrl");

async function loadSettings() {
  try {
    const result = await chrome.storage.local.get(["backend_url"]);
    const url = result.backend_url || DEFAULT_BACKEND_URL;
    backendUrlInput.value = url;
    currentUrlEl.textContent = `Current: ${url}`;
  } catch (error) {
    console.error("Failed to load settings:", error);
    backendUrlInput.value = DEFAULT_BACKEND_URL;
    currentUrlEl.textContent = `Current: ${DEFAULT_BACKEND_URL} (default)`;
  }
}

async function saveSettings() {
  const url = backendUrlInput.value.trim();
  if (!url) {
    showTestResult("Please enter a valid URL", "error");
    return;
  }

  try {
    await chrome.storage.local.set({ backend_url: url });
    currentUrlEl.textContent = `Current: ${url}`;
    showTestResult("Settings saved successfully", "success");
  } catch (error) {
    console.error("Failed to save settings:", error);
    showTestResult("Failed to save settings", "error");
  }
}

async function testConnection() {
  const url = backendUrlInput.value.trim();
  if (!url) {
    showTestResult("Please enter a valid URL first", "error");
    return;
  }

  testConnectionBtn.disabled = true;
  testConnectionBtn.textContent = "Testing...";
  showTestResult("Testing connection...", "info");

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(`${url}/api/health`, {
      method: "GET",
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      showTestResult(`Connection successful! API status: ${data.status}`, "success");
      await saveSettings();
    } else {
      showTestResult(`Connection failed: HTTP ${response.status}`, "error");
    }
  } catch (error) {
    if (error.name === "AbortError") {
      showTestResult("Connection timed out (10s)", "error");
    } else {
      showTestResult(`Connection error: ${error.message}`, "error");
    }
  } finally {
    testConnectionBtn.disabled = false;
    testConnectionBtn.textContent = "Test API Connection";
  }
}

function showTestResult(message, type) {
  testResult.textContent = message;
  testResult.className = `help-text ${type}`;
}

backendUrlInput.addEventListener("change", saveSettings);
backendUrlInput.addEventListener("blur", saveSettings);
testConnectionBtn.addEventListener("click", testConnection);

document.addEventListener("DOMContentLoaded", loadSettings);