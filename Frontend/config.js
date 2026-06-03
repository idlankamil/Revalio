// ============================================
// Revalio — Central API Config
// ============================================
// Change ONLY this line when you get your ngrok URL!
window.REVALIO_API_BASE = "https://intertissued-cinthia-weer.ngrok-free.dev";

// Auto-inject ngrok bypass header for all fetch calls
const _originalFetch = window.fetch;
window.fetch = function(url, options = {}) {
  if (typeof url === "string" && url.includes("ngrok")) {
    options.headers = {
      ...(options.headers || {}),
      "ngrok-skip-browser-warning": "true"
    };
  }
  return _originalFetch(url, options);
};

// Fix image_url that comes from DB as "http://localhost:8000/images/..."
// Replace localhost:8000 with the real API base
window.fixImageUrl = function(url) {
  if (!url) return null;
  if (url.includes('localhost:8000')) {
    return url.replace('http://localhost:8000', window.REVALIO_API_BASE);
  }
  if (url.includes('127.0.0.1:8000')) {
    return url.replace('http://127.0.0.1:8000', window.REVALIO_API_BASE);
  }
  return url;
};