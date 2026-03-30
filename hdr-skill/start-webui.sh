#!/bin/bash

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WEBUI_DIR="$SCRIPT_DIR/webui"

echo "🚀 Starting HDR WebUI..."

# Check if backend dependencies are installed
if [ ! -d "$WEBUI_DIR/backend/node_modules" ]; then
  echo "📦 Installing backend dependencies..."
  cd "$WEBUI_DIR/backend"
  npm install
fi

# Check if frontend dependencies are installed
if [ ! -d "$WEBUI_DIR/frontend/node_modules" ]; then
  echo "📦 Installing frontend dependencies..."
  cd "$WEBUI_DIR/frontend"
  npm install
fi

# Build frontend
echo "🏗️  Building frontend..."
cd "$WEBUI_DIR/frontend"
npm run build

# Start backend
echo "🔧 Starting server..."
cd "$WEBUI_DIR/backend"
npm start
