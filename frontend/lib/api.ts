// API service for communicating with the crop disease detection backend

// NEXT_PUBLIC_* values are inlined into the client bundle at build time, so this
// must be set when `next build` runs — setting it on the running Worker has no
// effect. Falls back to the local backend for `next dev`.
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "")

export interface HealthResponse {
  status: string
  device: string
  model_loaded: boolean
  model_path: string
}

export interface PredictResponse {
  class: string
  confidence: number
  description: string
  treatment: string
  /** Shannon entropy of the class distribution; high values mean the model was unsure. */
  entropy?: number
}

export interface ApiError {
  message: string
  status?: number
}

const handleApiError = (error: unknown, defaultMsg: string): ApiError => ({
  message: error instanceof Error ? error.message : defaultMsg,
  status: 0,
})

// Health check endpoint
export async function checkHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    })
    if (!response.ok) throw new Error(`Health check failed with status ${response.status}`)
    return response.json()
  } catch (error) {
    console.error("Health check error:", error)
    throw handleApiError(error, "Failed to connect to backend")
  }
}

// Prediction endpoint
export async function predictDisease(file: File): Promise<PredictResponse> {
  try {
    const formData = new FormData()
    formData.append("file", file)
    const response = await fetch(`${API_BASE_URL}/predict`, { method: "POST", body: formData })
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `Prediction failed with status ${response.status}`)
    }
    return response.json()
  } catch (error) {
    console.error("Prediction error:", error)
    throw handleApiError(error, "Failed to analyze image")
  }
}
