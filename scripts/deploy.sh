#!/bin/bash
set -e

APP_USER="eduwhistle-chatbot"
PROJECT_DIR="/home/eduwhistle-chatbot/htdocs/chatbot.eduwhistle.com/chatbot"

echo "🔄 Switching to app user..."
cd $PROJECT_DIR

# Ensure correct ownership (idempotent safety)
echo "🔐 Fixing ownership..."
chown -R $APP_USER:$APP_USER $PROJECT_DIR

echo "🔄 Resetting repo..."
sudo -u $APP_USER git reset --hard

echo "⬇️ Pulling latest code..."
sudo -u $APP_USER git pull origin main

# Backend
echo "⚙️ Backend setup..."
cd backend
sudo -u $APP_USER bash -c "source venv/bin/activate && pip install -r requirements.txt"

echo "🔁 Restarting backend service..."
systemctl restart chatbot.service

# Frontend
echo "🎨 Frontend build..."
cd ../frontend

# Clean old build (important)
sudo -u $APP_USER rm -rf dist

# Install & build
sudo -u $APP_USER npm ci
sudo -u $APP_USER npm run build

# Build widget and copy to dist
echo "🎨 Widget build..."
sudo -u $APP_USER npm run build:widget
sudo -u $APP_USER cp dist-widget/widget.js dist/

echo "✅ Deployment complete"