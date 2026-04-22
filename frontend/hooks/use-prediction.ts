import { useCallback, useState, useEffect } from "react"
import { checkHealth, predictDisease, type HealthResponse, type PredictResponse, type ApiError } from "@/lib/api"

export interface usePredictionState {
  isHealthy: boolean
  isLoading: boolean
  isPredicting: boolean
  prediction: PredictResponse | null
  error: ApiError | null
  healthData: HealthResponse | null
}

const initialState: usePredictionState = {
  isHealthy: false,
  isLoading: true,
  isPredicting: false,
  prediction: null,
  error: null,
  healthData: null,
}

export function usePrediction() {
  const [state, setState] = useState<usePredictionState>(initialState)

  useEffect(() => {
    checkHealth()
      .then((health) => setState((prev) => ({
        ...prev,
        isHealthy: health.status === "ok",
        isLoading: false,
        healthData: health,
        error: null,
      })))
      .catch((error) => setState((prev) => ({
        ...prev,
        isHealthy: false,
        isLoading: false,
        error: error as ApiError,
      })))
  }, [])

  const runPrediction = useCallback(async (file: File) => {
    setState((prev) => ({ ...prev, isPredicting: true, error: null, prediction: null }))
    try {
      const result = await predictDisease(file)
      setState((prev) => ({ ...prev, isPredicting: false, prediction: result, error: null }))
      return result
    } catch (error) {
      setState((prev) => ({ ...prev, isPredicting: false, error: error as ApiError, prediction: null }))
      throw error
    }
  }, [])

  return { ...state, runPrediction }
}
