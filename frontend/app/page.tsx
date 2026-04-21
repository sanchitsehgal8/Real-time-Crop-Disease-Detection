"use client"

import { useState, useCallback } from "react"
import { Header } from "@/components/header"
import { Footer } from "@/components/footer"
import { SpecimenInput } from "@/components/specimen-input"
import { DiagnosticPreview } from "@/components/diagnostic-preview"
import { Button } from "@/components/ui/button"
import { BarChart3, AlertCircle } from "lucide-react"
import { usePrediction } from "@/hooks/use-prediction"

export default function CropDiseaseAnalysis() {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const { isHealthy, isPredicting, prediction, error, isLoading, runPrediction } = usePrediction()

  const handleImageUpload = useCallback((file: File) => {
    setUploadedFile(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }, [])

  const handleRunPrediction = async () => {
    if (!uploadedFile) return

    try {
      await runPrediction(uploadedFile)
    } catch (err) {
      console.error("Prediction failed:", err)
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      
      <main className="flex-1 px-6 py-8 md:px-12 lg:px-20">
        <div className="max-w-6xl mx-auto">
          {/* Backend Status Alert */}
          {isLoading ? (
            <div className="bg-blue-50 border border-blue-200 rounded-2xl p-6 mb-8 flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              <span className="text-blue-700 text-sm font-medium">Connecting to backend...</span>
            </div>
          ) : !isHealthy ? (
            <div className="bg-red-50 border border-red-200 rounded-2xl p-6 mb-8 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-red-700 font-medium">Backend Unavailable</p>
                <p className="text-red-600 text-sm mt-1">
                  {error?.message || "Unable to connect to the crop disease detection service. Please ensure the backend is running on http://127.0.0.1:8000"}
                </p>
              </div>
            </div>
          ) : (
            <div className="bg-green-50 border border-green-200 rounded-2xl p-6 mb-8 flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-green-700 text-sm font-medium">Backend connected and ready</span>
            </div>
          )}

          <div className="bg-card rounded-2xl p-8 md:p-12 shadow-sm">
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-2">
              Crop Disease Analysis
            </h1>
            <p className="text-muted-foreground mb-8">
              Advanced specimen diagnostics for early detection and mitigation.
            </p>

            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <SpecimenInput 
                onImageUpload={handleImageUpload} 
                uploadedImage={uploadedImage}
                isBackendReady={isHealthy}
              />
              <DiagnosticPreview 
                uploadedImage={uploadedImage}
                isAnalyzing={isPredicting}
                prediction={prediction}
                error={error}
              />
            </div>

            {error && !isHealthy && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6">
                <p className="text-red-700 text-sm">{error.message}</p>
              </div>
            )}

            <div className="flex justify-center">
              <Button 
                onClick={handleRunPrediction}
                disabled={!uploadedFile || isPredicting || !isHealthy || isLoading}
                className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-6 text-sm font-semibold tracking-wide"
              >
                <BarChart3 className="w-4 h-4 mr-2" />
                {isPredicting ? "ANALYZING..." : "RUN PREDICTION MODEL"}
              </Button>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}
