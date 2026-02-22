#!/bin/bash
# Quick start script for Open-Sable

echo "🚀 Open-Sable Quick Start"
echo "========================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run: python install.py"
    exit 1
fi

# Activate venv
source venv/bin/activate

case "$1" in
    whatsapp)
        echo "📱 Starting WhatsApp Bridge..."
        cd whatsapp-bridge && node bridge.js
        ;;
    skills)
        echo "🛒 Opening Skills Hub..."
        python sable.py skills
        ;;
    onboarding)
        echo "🎯 Running Onboarding Wizard..."
        python sable.py onboarding
        ;;
    test)
        echo "🧪 Running Tests..."
        python -m pytest tests/test_features.py tests/test_unit.py -v
        ;;
    chat)
        echo "💬 Starting CLI Chat..."
        CLI_ENABLED=true python -m opensable
        ;;
    *)
        echo "📋 Starting Open-Sable agent..."
        echo ""
        echo "  🌐 WebChat will be at http://127.0.0.1:8789"
        echo "  + Any configured platforms (Telegram, Discord, WhatsApp, Slack)"
        echo ""
        python -m opensable
        ;;
esac
