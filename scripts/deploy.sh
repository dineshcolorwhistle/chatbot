#!/bin/bash
set -e

PROJECT_DIR="/home/eduwhistle-chatbot/htdocs/chatbot.eduwhistle.com/chatbot"

cd $PROJECT_DIR

echo "🔄 Resetting repo..."
git reset --hard

echo "⬇️ Pulling latest code..."
git pull origin main

# Backend
cd backend
source venv/bin/activate
pip install -r requirements.txt

sudo systemctl restart chatbot.service

# Frontend
cd ../frontend
npm ci
npm run build

echo "✅ Deployment complete"