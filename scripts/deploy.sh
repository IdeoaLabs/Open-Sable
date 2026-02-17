#!/bin/bash
# Production deployment script

set -e

echo "🚀 Deploying Open-Sable to production..."

# Update code
echo "📥 Pulling latest code..."
git pull origin main

# Install dependencies
echo "📦 Installing dependencies..."
python3 -m pip install -r requirements.txt --upgrade

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v

# Restart service
echo "🔄 Restarting service..."
sudo systemctl restart opensable

# Check status
echo "✅ Checking status..."
sudo systemctl status opensable --no-pager

echo ""
echo "✅ Deployment complete!"
echo "View logs: journalctl -u opensable -f"
