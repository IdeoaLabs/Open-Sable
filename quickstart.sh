#!/bin/bash
set -e  # Exit on error

echo "🚀 Open-Sable Quick Setup for Linux/Mac"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]); then
    echo "❌ Python 3.11+ required. Current: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create venv
if [ ! -d "venv" ]; then
    echo ""
    echo "🔨 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate venv
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install core
echo ""
echo "📦 Installing Open-Sable (this may take a few minutes)..."
pip install -r requirements.txt
pip install -e ".[core]"

# Create directories
echo ""
echo "📁 Setting up directories..."
mkdir -p data logs config
echo "✅ Directories created"

# Setup .env
echo ""
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "⚙️  Setting up configuration..."
        cp .env.example .env
        echo "✅ Created .env file"
    else
        echo "⚠️  .env.example not found - skipping .env creation"
    fi
else
    echo "✅ .env file already exists"
fi

# Install Ollama automatically if not present
echo ""
if ! command -v ollama &> /dev/null; then
    echo "📥 Ollama not found - installing automatically..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "❌ Homebrew not found - installing Ollama manually..."
            curl -fsSL https://ollama.com/install.sh | sh
        fi
    else
        # Linux
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    
    echo "✅ Ollama installed"
else
    echo "✅ Ollama is already installed"
fi

# Pull model automatically based on system specs
echo ""
echo "🔍 Detecting system specifications..."

# Get RAM in GB and CPU cores
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
CPU_CORES=$(nproc)
echo "   RAM: ${RAM_GB}GB | CPU Cores: $CPU_CORES"

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_MEM_GB=$((GPU_MEM / 1024))
    echo "   GPU: NVIDIA with ${GPU_MEM_GB}GB VRAM"
    HAS_GPU=true
else
    echo "   GPU: None detected (CPU only)"
    HAS_GPU=false
fi

# Auto-select optimal model based on hardware
if [ "$HAS_GPU" = true ] && [ "$GPU_MEM_GB" -ge 20 ]; then
    MODEL="llama3.1:70b"
    echo "   🚀 Auto-selected: $MODEL (Best quality for your GPU)"
elif [ "$HAS_GPU" = true ] && [ "$GPU_MEM_GB" -ge 8 ]; then
    MODEL="llama3.1:8b"
    echo "   🚀 Auto-selected: $MODEL (Balanced GPU performance)"
elif [ "$RAM_GB" -ge 32 ]; then
    MODEL="llama3.1:8b"
    echo "   🚀 Auto-selected: $MODEL (Balanced performance)"
elif [ "$RAM_GB" -ge 8 ]; then
    MODEL="llama3.2:3b"
    echo "   🚀 Auto-selected: $MODEL (Efficient)"
else
    MODEL="llama3.2:1b"
    echo "   🚀 Auto-selected: $MODEL (Ultra efficient)"
fi

echo "   📦 Agent will auto-download additional models as needed during runtime"

echo ""
if ollama list 2>/dev/null | grep -q "$MODEL"; then
    echo "✅ $MODEL already installed"
else
    echo "📥 Downloading $MODEL (this may take a while)..."
    ollama pull "$MODEL"
    echo "✅ Model downloaded"
fi

# Update .env with selected model
if [ -f .env ]; then
    sed -i "s/DEFAULT_MODEL=.*/DEFAULT_MODEL=$MODEL/" .env
    echo "✅ Updated .env with model: $MODEL"
fi

echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║                                           ║"
echo "║   ✅ Installation Complete!               ║"
echo "║                                           ║"
echo "╚═══════════════════════════════════════════╝"
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Edit .env with your bot token:"
echo "   - Get from @BotFather on Telegram"
echo "   - Set: TELEGRAM_BOT_TOKEN=your_token_here"
echo ""
echo "3. Start Open-Sable:"
echo "   python -m opensable"
echo ""
echo "📚 More info: README.md | INSTALL.md"
echo ""
