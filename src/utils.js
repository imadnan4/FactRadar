// Utility functions for FactRadar

export function formatConfidence(confidence) {
  return (confidence * 100).toFixed(2) + "%"
}

export function getConfidenceLevel(confidence) {
  if (confidence >= 0.8) return "High"
  if (confidence >= 0.6) return "Medium"
  return "Low"
}

export function sanitizeInput(text) {
  return text.trim().replace(/\s+/g, " ")
}

export function validateInput(text) {
  if (!text || text.trim().length === 0) {
    return { valid: false, error: "Text cannot be empty" }
  }
  if (text.length > 1000) {
    return { valid: false, error: "Text too long (max 1000 characters)" }
  }
  return { valid: true }
}

export function debounce(func, wait) {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}
