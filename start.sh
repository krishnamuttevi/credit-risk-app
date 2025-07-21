#!/bin/bash

echo ""
echo "🚀 Starting Credit Risk App"
echo "🔍 Main App (index.html): http://127.0.0.1:9000/"
echo "📘 API Docs (Swagger): http://127.0.0.1:9000/docs"
echo ""

# Start the FastAPI server
exec uvicorn main:app --host 0.0.0.0 --port 9000
