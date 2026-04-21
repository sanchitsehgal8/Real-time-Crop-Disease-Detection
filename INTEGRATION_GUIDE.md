# Complete Backend-Frontend Integration Guide

This document explains how to run the fully connected crop disease detection system.

## Prerequisites

- Python 3.8+
- Node.js 18+ with npm
- The YOLOv8 model file (`best.pt`) in the project root
- Backend dependencies (see `requirements.txt`)

## Backend Setup & Running

### 1. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend

```bash
python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

The backend will be available at: **http://127.0.0.1:8000**

**Available endpoints:**
- `GET /health` - Health check endpoint
- `POST /predict` - Crop disease prediction endpoint

You can also access the **interactive API documentation** at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Frontend Setup & Running

### 1. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 2. Start the Next.js Development Server

```bash
npm run dev
```

The frontend will be available at: **http://localhost:3000**

## End-to-End Integration Workflow

1. **Backend Health Check**: The frontend automatically checks the backend health on load
   - If backend is running: Shows green "Backend connected and ready" badge
   - If backend is down: Shows red alert with connection error

2. **Image Upload**: 
   - Drag and drop or click to upload a crop leaf image (JPG/PNG, max 50MB)
   - Supported formats: JPEG, PNG, TIFF

3. **Disease Prediction**:
   - Click "RUN PREDICTION MODEL" button
   - Frontend sends the image to the backend via `/predict` endpoint
   - Backend performs inference using YOLOv8 model

4. **Results Display**:
   - Prediction shows:
     - Detected crop disease/plant type
     - Confidence percentage
     - Agricultural description of the disease
     - Treatment recommendations

## Features Implemented

### Backend (FastAPI)
- ✅ CORS middleware configured for cross-origin requests
- ✅ `/health` endpoint for backend status
- ✅ `/predict` endpoint for crop disease classification
- ✅ Disease information database with 16 crop/disease combinations
- ✅ Treatment recommendations for each disease
- ✅ Proper error handling and validation

### Frontend (Next.js)
- ✅ API service layer for backend communication
- ✅ Health check on application load
- ✅ File upload with validation
- ✅ Real-time prediction with loading states
- ✅ Results display with disease description and treatment info
- ✅ Error handling with user-friendly messages
- ✅ Backend status indicator
- ✅ Responsive UI design

## API Integration Details

### CORS Configuration
The backend allows requests from:
- `http://localhost:3000`
- `http://127.0.0.1:3000`
- `http://localhost:5173` (alternative dev port)
- `http://127.0.0.1:5173`
- `http://localhost:8000`
- `http://127.0.0.1:8000`

### Health Check Response
```json
{
  "status": "ok",
  "device": "cuda|mps|cpu",
  "model_loaded": true,
  "model_path": "best.pt"
}
```

### Prediction Request
- **Method**: POST
- **URL**: `/predict`
- **Content-Type**: multipart/form-data
- **Body**: `file` (binary image data)

### Prediction Response
```json
{
  "class": "Apple___Apple_scab",
  "confidence": 0.9876,
  "description": "Fungal disease that causes olive-brown lesions on leaves and fruit, reducing fruit quality.",
  "treatment": "Prune infected foliage, improve airflow, and apply preventive fungicides during wet periods."
}
```

## Troubleshooting

### Backend not connecting?
1. Check if backend is running on `http://127.0.0.1:8000`
2. Verify Python virtual environment is activated
3. Ensure `requirements.txt` dependencies are installed
4. Check model file (`best.pt`) exists in project root

### Frontend shows "Backend Unavailable"?
1. Start the backend server (see Backend Setup)
2. Wait 5 seconds for backend to load model
3. Refresh the frontend page
4. Check browser console for detailed error messages

### Frontend not loading?
1. Ensure Node.js 18+ is installed (`node --version`)
2. Delete `frontend/node_modules` and `frontend/.next`
3. Run `npm install` again
4. Verify Next.js dev server is running on port 3000

### Upload button disabled?
- Wait for backend health check to complete
- Ensure an image is selected
- Verify backend status shows as connected (green badge)

## Project Structure

```
Real-time-Crop-Disease-Detection/
├── backend/
│   ├── app.py              # FastAPI application with CORS
│   ├── model.py            # YOLOv8 model handling
│   └── __init__.py
├── frontend/
│   ├── app/
│   │   ├── page.tsx        # Main page with integration
│   │   └── layout.tsx
│   ├── components/
│   │   ├── diagnostic-preview.tsx  # Results display
│   │   ├── specimen-input.tsx      # Upload component
│   │   ├── header.tsx
│   │   └── footer.tsx
│   ├── lib/
│   │   ├── api.ts          # Backend API service layer
│   │   └── utils.ts
│   ├── hooks/
│   │   └── use-prediction.ts  # Prediction state management
│   └── package.json
├── best.pt                 # YOLOv8 model file
├── requirements.txt        # Python dependencies
└── data/                   # Training/validation data
```

## Next Steps for Production

1. **Environment Variables**:
   - Create `.env.local` in frontend for API base URL configuration
   - Use environment variables for backend port/host

2. **Build for Production**:
   ```bash
   # Backend
   # Just run with gunicorn/production ASGI server
   
   # Frontend
   cd frontend
   npm run build
   npm start
   ```

3. **Deployment**:
   - Deploy backend to cloud platform (Heroku, AWS, GCP, etc.)
   - Deploy frontend to Vercel, Netlify, or self-hosted
   - Update CORS origins in backend for production domains

4. **Security**:
   - Implement authentication for API endpoints
   - Use HTTPS in production
   - Validate file uploads server-side
   - Implement rate limiting

---

For questions or issues, check the browser console (F12) and backend terminal for detailed error messages.
