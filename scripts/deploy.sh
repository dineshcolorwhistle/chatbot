#!/bin/bash

set -e  # stop on error

echo "🚀 Starting deployment..."

PROJECT_DIR="/home/eduwhistle-aichat/htdocs/aichat.eduwhistle.com/chatbot"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

cd $PROJECT_DIR

echo "🔄 Resetting repo..."
git fetch origin
git reset --hard origin/main
git clean -fd

echo "⬇️ Latest code synced"

# -----------------------
# Backend Setup
# -----------------------
echo "⚙️ Backend setup..."

cd $BACKEND_DIR

# Ensure venv exists
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# -----------------------
# Restart Backend
# -----------------------
echo "🔁 Restarting backend service..."
sudo systemctl restart aichat

# Verify service
sleep 2
if ! systemctl is-active --quiet aichat; then
  echo "❌ Backend failed to start"
  journalctl -u aichat -n 50 --no-pager
  exit 1
fi

echo "✅ Backend is running"

# -----------------------
# Frontend Setup
# -----------------------
echo "🎨 Frontend build..."

cd $FRONTEND_DIR

echo "🧹 Cleaning old build..."
rm -rf dist dist-widget

# Optional: ensure correct node version
# source ~/.nvm/nvm.sh
# nvm use 18

echo "📦 Installing dependencies..."
npm ci

echo "🏗️ Building frontend..."
npm run build

echo "🧩 Building widget..."
npm run build:widget

# Ensure widget copy
if [ -f "dist-widget/widget.js" ]; then
  cp dist-widget/widget.js dist/
else
  echo "❌ widget.js not found!"
  exit 1
fi

echo "✅ Frontend build complete"

# -----------------------
# Done
# -----------------------
echo "🎉 Deployment completed successfully!"