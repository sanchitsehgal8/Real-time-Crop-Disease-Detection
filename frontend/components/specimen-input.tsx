"use client"

import { useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { CloudUpload } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface SpecimenInputProps {
  onImageUpload: (file: File) => void
  uploadedImage: string | null
}

export function SpecimenInput({ onImageUpload, uploadedImage }: SpecimenInputProps) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => acceptedFiles.length > 0 && onImageUpload(acceptedFiles[0]),
    accept: { 'image/jpeg': ['.jpg', '.jpeg'], 'image/png': ['.png'], 'image/tiff': ['.tiff', '.tif'] },
    maxSize: 50 * 1024 * 1024,
    multiple: false
  })

  return (
    <div className="bg-white rounded-xl p-6 border border-border">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-foreground">Specimen Input</h2>
        <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
          <span className="w-2 h-2 rounded-full bg-green-500 mr-2" />
          READY
        </Badge>
      </div>

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors duration-200 ${isDragActive ? 'border-primary bg-primary/5' : 'border-gray-300 hover:border-primary/50'} ${uploadedImage ? 'border-green-400 bg-green-50' : ''}`}
      >
        <input {...getInputProps()} />
        {uploadedImage ? (
          <div className="space-y-3">
            <img src={uploadedImage} alt="Uploaded specimen" className="max-h-32 mx-auto rounded-lg object-contain" />
            <p className="text-sm text-muted-foreground">Click or drag to replace image</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="w-14 h-14 rounded-full bg-green-100 flex items-center justify-center mx-auto">
              <CloudUpload className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <p className="font-medium text-foreground">Drag and drop specimen image</p>
              <p className="text-sm text-muted-foreground">or click to browse local files</p>
            </div>
            <p className="text-xs text-muted-foreground">Supports JPEG, PNG, TIFF up to 50MB</p>
          </div>
        )}
      </div>
    </div>
  )
}
