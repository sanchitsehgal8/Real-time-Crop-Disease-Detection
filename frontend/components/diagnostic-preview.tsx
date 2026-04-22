"use client"

import { Microscope, Loader2, AlertCircle, CheckCircle } from "lucide-react"
import { type PredictResponse, type ApiError } from "@/lib/api"

interface DiagnosticPreviewProps {
  uploadedImage: string | null
  isAnalyzing: boolean
  prediction: PredictResponse | null
  error: ApiError | null
}

const InfoRow = ({ label, value }: { label: string; value: string }) => (
  <div>
    <p className="text-xs font-semibold text-gray-600 mb-1">{label}</p>
    <p className="text-sm text-foreground">{value}</p>
  </div>
)

const CenterContent = ({ Icon, title, subtitle }: any) => (
  <div className="text-center space-y-3 w-full">
    <Icon className="w-10 h-10 text-gray-300 mx-auto" />
    <p className="text-sm text-muted-foreground max-w-[200px]">{title}</p>
    {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
  </div>
)

export function DiagnosticPreview({ uploadedImage, isAnalyzing, prediction, error }: DiagnosticPreviewProps) {
  const isHealthy = prediction?.class.toLowerCase().includes("healthy")

  const renderContent = () => {
    if (isAnalyzing) return <CenterContent Icon={Loader2} title="Analyzing specimen..." />
    if (error) return (
      <div className="text-center space-y-3 w-full">
        <AlertCircle className="w-10 h-10 text-red-500 mx-auto" />
        <p className="text-sm text-red-600 font-medium">Analysis failed</p>
        <p className="text-xs text-red-500 max-w-[280px]">{error.message}</p>
      </div>
    )
    if (prediction) return (
      <div className="w-full text-center space-y-4">
        {uploadedImage && <img src={uploadedImage} alt="Analyzed" className="max-h-40 mx-auto rounded-lg object-contain" />}
        <div className="flex items-center justify-center gap-2">
          {isHealthy ? <CheckCircle className="w-5 h-5 text-green-600" /> : <AlertCircle className="w-5 h-5 text-amber-600" />}
          <p className="font-semibold text-foreground">{prediction.class}</p>
        </div>
        <div className="inline-block bg-primary/10 px-3 py-1 rounded-full">
          <p className="text-sm font-medium text-primary">{(prediction.confidence * 100).toFixed(1)}% confidence</p>
        </div>
        <div className="text-left space-y-3 bg-gray-50 p-4 rounded-lg">
          <InfoRow label="DESCRIPTION" value={prediction.description} />
          <InfoRow label="TREATMENT" value={prediction.treatment} />
        </div>
      </div>
    )
    if (uploadedImage) return (
      <div className="text-center space-y-3">
        <img src={uploadedImage} alt="Preview" className="max-h-32 mx-auto rounded-lg object-contain" />
        <p className="text-sm text-muted-foreground">Ready for analysis</p>
      </div>
    )
    return <CenterContent Icon={Microscope} title="Upload a specimen image to generate a detailed diagnostic report." />
  }

  return (
    <div className="bg-white rounded-xl p-6 border border-border">
      <h2 className="text-lg font-semibold text-foreground mb-4">Diagnostic Preview</h2>
      <div className="border border-gray-200 rounded-lg p-8 min-h-[300px] flex flex-col items-center justify-center bg-gray-50/50">
        {renderContent()}
      </div>
    </div>
  )
}
