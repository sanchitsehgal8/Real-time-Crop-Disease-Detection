#!/bin/bash

# Integration Verification Script
# This script checks if all components are properly set up

echo "🔍 Checking Backend-Frontend Integration Setup..."
echo ""

# Check Python
echo "1️⃣  Checking Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "   ✅ $PYTHON_VERSION"
else
    echo "   ❌ Python 3 not found - Install Python 3.8+"
fi

# Check Node.js
echo "2️⃣  Checking Node.js environment..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    NPM_VERSION=$(npm --version)
    echo "   ✅ Node.js $NODE_VERSION"
    echo "   ✅ npm $NPM_VERSION"
else
    echo "   ❌ Node.js not found - Install Node.js 18+"
fi

# Check backend files
echo "3️⃣  Checking backend files..."
if [ -f "backend/app.py" ]; then
    echo "   ✅ backend/app.py exists"
else
    echo "   ❌ backend/app.py not found"
fi

if [ -f "requirements.txt" ]; then
    echo "   ✅ requirements.txt exists"
else
    echo "   ❌ requirements.txt not found"
fi

# Check model file
echo "4️⃣  Checking YOLOv8 model..."
if [ -f "best.pt" ]; then
    echo "   ✅ best.pt model file found"
else
    echo "   ❌ best.pt model file not found - Download or train the model"
fi

# Check frontend files
echo "5️⃣  Checking frontend files..."
if [ -f "frontend/package.json" ]; then
    echo "   ✅ frontend/package.json exists"
else
    echo "   ❌ frontend/package.json not found"
fi

if [ -f "frontend/app/page.tsx" ]; then
    echo "   ✅ frontend/app/page.tsx exists"
else
    echo "   ❌ frontend/app/page.tsx not found"
fi

if [ -f "frontend/lib/api.ts" ]; then
    echo "   ✅ frontend/lib/api.ts (API service layer) exists"
else
    echo "   ❌ frontend/lib/api.ts not found"
fi

if [ -f "frontend/hooks/use-prediction.ts" ]; then
    echo "   ✅ frontend/hooks/use-prediction.ts (prediction hook) exists"
else
    echo "   ❌ frontend/hooks/use-prediction.ts not found"
fi

# Check for CORS configuration
echo "6️⃣  Checking CORS configuration..."
if grep -q "CORSMiddleware" "backend/app.py"; then
    echo "   ✅ CORS middleware configured in backend"
else
    echo "   ❌ CORS middleware not found in backend"
fi

echo ""
echo "📋 Setup Checklist Summary:"
echo "   - Start backend: python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8000"
echo "   - Start frontend: cd frontend && npm install && npm run dev"
echo "   - Backend API: http://127.0.0.1:8000"
echo "   - Frontend URL: http://localhost:3000"
echo ""
echo "✨ Integration setup verification complete!"
