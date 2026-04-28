#!/bin/bash
set -e

PROJECT_DIR="/home/eduwhistle-chatbot/htdocs/chatbot.eduwhistle.com/chatbot"

cd $PROJECT_DIR

echo "🔄 Resetting repo..."
git reset --hard

echo "⬇️ Pulling latest code..."
git pull origin main

# Backend
echo "⚙️ Backend setup..."
cd backend
source venv/bin/activate
pip install -r requirements.txt

echo "🔁 Restarting backend service..."
sudo systemctl restart chatbot.service

# Frontend
cd ../frontend

echo "🧹 Cleaning old build..."
rm -rf dist

echo "📦 Installing deps..."
npm ci

echo "🏗️ Building..."
npm run build

echo "🎨 Widget build..."
npm run build:widget
cp dist-widget/widget.js dist/

echo "✅ Deployment complete"