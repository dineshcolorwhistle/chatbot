#!/bin/bash

set -e

PROJECT_DIR="/home/eduwhistle-chatbot/htdocs/chatbot.eduwhistle.com/chatbot"

echo "🚀 Deploy started at $(date)"

cd $PROJECT_DIR

git pull origin main

# Backend
cd backend
source venv/bin/activate
pip install -r requirements.txt

systemctl restart chatbot

# Frontend
cd ../frontend
npm ci
npm run build

echo "✅ Deploy completed"