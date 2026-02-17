#!/bin/bash
# Quick start script for SableCore

echo "🚀 SableCore Quick Start"
echo "======================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run: python install.py"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check what to start
if [ "$1" == "whatsapp" ]; then
    echo "📱 Starting WhatsApp Bridge..."
    cd venom-bot && node bridge.js
elif [ "$1" == "skills" ]; then
    echo "🛒 Opening Skills Hub..."
    python sable.py skills
elif [ "$1" == "onboarding" ]; then
    echo "🎯 Running Onboarding Wizard..."
    python sable.py onboarding
elif [ "$1" == "web" ]; then
    echo "🌐 Starting Web Dashboard..."
    python main.py --web
    echo "Open: http://localhost:8080/dashboard_modern.html"
elif [ "$1" == "test" ]; then
    echo "🧪 Running Tests..."
    python tests/test_features.py
elif [ "$1" == "chat" ]; then
    echo "💬 Starting CLI Chat..."
    python cli.py chat
else
    echo "📋 Starting main bot..."
    echo ""
    echo "Available platforms:"
    echo "  - Telegram (if TELEGRAM_BOT_TOKEN set)"
    echo "  - WhatsApp (if whatsapp_enabled=true)"
    echo ""
    python main.py
fi
