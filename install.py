#!/usr/bin/env python3
"""
Open-Sable Installation Script
Automated one-click installer for all platforms
"""
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


def print_banner():
    """Print installation banner"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🚀 Open-Sable Automated Installer                       ║
║   Your personal AI that actually does things              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
""")


def check_python_version():
    """Ensure Python 3.11+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python 3.11 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor} detected")


def install_ollama():
    """Install Ollama automatically if not present"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is already installed")
            return True
    except FileNotFoundError:
        pass
    
    print("📥 Ollama not found - installing automatically...")
    os_type = platform.system()
    
    try:
        if os_type == "Darwin":
            print("   Installing via Homebrew...")
            subprocess.run(['brew', 'install', 'ollama'], check=True)
        elif os_type == "Linux":
            print("   Installing via official script...")
            subprocess.run(['curl', '-fsSL', 'https://ollama.com/install.sh'], 
                         stdout=subprocess.PIPE, check=True)
            subprocess.run(['sh'], stdin=subprocess.PIPE, check=True)
        elif os_type == "Windows":
            print("❌ Cannot auto-install on Windows")
            print("   Please download from: https://ollama.com/download")
            return False
        
        print("✅ Ollama installed successfully")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Failed to install Ollama automatically")
        print(f"   Please install manually from: https://ollama.com")
        return False


def create_venv():
    """Create virtual environment if not exists"""
    venv_path = Path('venv')
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("\n🔨 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False


def get_venv_python():
    """Get path to venv Python executable"""
    if platform.system() == "Windows":
        return Path('venv') / 'Scripts' / 'python.exe'
    else:
        return Path('venv') / 'bin' / 'python'


def install_dependencies():
    """Install Python dependencies in venv"""
    print("\n📦 Installing dependencies...")
    
    venv_python = get_venv_python()
    
    try:
        # Upgrade pip first
        print("   Upgrading pip...")
        subprocess.run(
            [str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'],
            check=True,
            capture_output=True
        )
        
        # Install package with core dependencies
        print("   Installing opensable[core]...")
        subprocess.run(
            [str(venv_python), '-m', 'pip', 'install', '-e', '.[core]'],
            check=True
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def install_playwright():
    """Install Playwright browsers"""
    print("\n🌐 Installing Playwright browsers...")
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'playwright', 'install', 'chromium'],
            check=True
        )
        print("✅ Playwright browsers installed")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Playwright browser installation failed (optional)")
        return False


def install_venom_bot():
    """Install Venom Bot for WhatsApp integration"""
    print("\n💬 Install Venom Bot for WhatsApp? (y/n): ", end="")
    choice = input().strip().lower()
    if choice != "y":
        print("⏭️  Skipping Venom Bot installation")
        return
    
    # Check if Node.js is installed
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Node.js is required for WhatsApp integration")
            print("   Install Node.js 16+ and run installer again")
            return
        print(f"✅ Node.js {result.stdout.strip()} detected")
    except FileNotFoundError:
        print("❌ Node.js not found - install it first")
        return
    
    venom_dir = Path("venom-bot")
    
    # Check if already exists
    if venom_dir.exists() and (venom_dir / "package.json").exists():
        print("✅ Venom Bot already installed")
        return
    
    print("📥 Installing Venom Bot dependencies...")
    
    try:
        # Install npm dependencies
        subprocess.run(
            ['npm', 'install', 'venom-bot', 'express', 'dotenv'],
            cwd=str(venom_dir),
            check=True
        )
        print("✅ Venom Bot installed successfully")
        print("\n📱 To connect WhatsApp:")
        print("   1. Run: cd venom-bot && node bridge.js")
        print("   2. Scan QR code with WhatsApp")
        print("   3. Start OpenSable with WHATSAPP_ENABLED=true")
    except subprocess.CalledProcessError:
        print("⚠️  Venom Bot installation failed - install manually")
    except FileNotFoundError:
        print("❌ npm not found - install Node.js and npm first")


def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    dirs = ['data', 'logs', 'config']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✅ Directories created")


def setup_env_file():
    """Create .env file from example"""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if env_file.exists():
        print("\n✅ .env file already exists")
        return
    
    if env_example.exists():
        print("\n⚙️  Setting up configuration...")
        
        # Copy example
        import shutil
        shutil.copy(env_example, env_file)
        
        print("✅ Created .env file from template")
        print("\n⚠️  IMPORTANT: Edit .env file to add your bot tokens:")
        print("   - TELEGRAM_BOT_TOKEN (get from @BotFather on Telegram)")
        print("   - DISCORD_BOT_TOKEN (get from Discord Developer Portal)")
    else:
        print("⚠️  .env.example not found")


def detect_hardware():
    """Detect system hardware specs"""
    import psutil
    
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_cores = psutil.cpu_count(logical=True)
    
    # Try to detect GPU
    gpu_mem_gb = 0
    has_gpu = False
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_mem_mb = int(result.stdout.strip().split('\n')[0])
            gpu_mem_gb = gpu_mem_mb / 1024
            has_gpu = True
    except:
        pass
    
    return {
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'gpu_mem_gb': gpu_mem_gb,
        'has_gpu': has_gpu
    }


def get_model_recommendations(specs):
    """Get model recommendations based on hardware"""
    models = []
    
    if specs['has_gpu'] and specs['gpu_mem_gb'] >= 20:
        models = [
            ("llama3.1:70b", "24GB VRAM - Best quality for high-end GPU"),
            ("llama3.1:8b", "5GB VRAM - Fast and reliable"),
            ("qwen2.5:7b", "4GB VRAM - Good reasoning"),
            ("llama3.2:3b", "2GB VRAM - Efficient"),
            ("gemma2:9b", "6GB VRAM - Fast reasoning")
        ]
    elif specs['has_gpu'] and specs['gpu_mem_gb'] >= 8:
        models = [
            ("llama3.1:8b", "5GB VRAM - Balanced GPU performance"),
            ("qwen2.5:7b", "4GB VRAM - Good reasoning"),
            ("gemma2:9b", "6GB VRAM - Fast reasoning"),
            ("llama3.2:3b", "2GB VRAM - Efficient"),
            ("phi3:14b", "8GB VRAM - Advanced reasoning")
        ]
    elif specs['ram_gb'] >= 32:
        models = [
            ("llama3.1:8b", "5GB RAM - Balanced"),
            ("qwen2.5:7b", "4GB RAM - Good reasoning"),
            ("gemma2:9b", "6GB RAM - Fast"),
            ("llama3.2:3b", "2GB RAM - Efficient"),
            ("phi3:14b", "8GB RAM - Advanced")
        ]
    elif specs['ram_gb'] >= 8:
        models = [
            ("llama3.2:3b", "2GB RAM - Fast and capable"),
            ("gemma2:2b", "2GB RAM - Efficient"),
            ("qwen2.5:3b", "2GB RAM - Good reasoning"),
            ("phi3:3.8b", "3GB RAM - Compact"),
            ("llama3.2:1b", "1GB RAM - Ultra fast")
        ]
    else:
        models = [
            ("llama3.2:1b", "1GB RAM - Ultra efficient"),
            ("qwen2.5:0.5b", "500MB RAM - Minimal"),
            ("gemma2:2b", "2GB RAM - Balanced")
        ]
    
    return models


def pull_ollama_model():
    """Pull optimal Ollama model based on system"""
    print("\n🤖 Detecting system for optimal model selection...")
    
    try:
        specs = detect_hardware()
        print(f"\n   RAM: {specs['ram_gb']:.1f}GB | CPU Cores: {specs['cpu_cores']}")
        if specs['has_gpu']:
            print(f"   GPU: NVIDIA with {specs['gpu_mem_gb']:.1f}GB VRAM")
        else:
            print("   GPU: None detected (CPU only)")
        
        models = get_model_recommendations(specs)
        
        # Auto-select optimal model (first in list)
        selected_model, desc = models[0]
        print(f"\n   🚀 Auto-selected: {selected_model} - {desc}")
        print("   📦 Agent will auto-download additional models as needed during runtime")
        
        # Check if already installed
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        
        if selected_model in result.stdout:
            print(f"✅ {selected_model} already installed")
        else:
            print(f"\n📥 Downloading {selected_model} (this may take a while)...")
            subprocess.run(['ollama', 'pull', selected_model], check=True)
            print("✅ Model downloaded")
        
        # Update .env with selected model
        env_file = Path('.env')
        if env_file.exists():
            content = env_file.read_text()
            import re
            content = re.sub(r'DEFAULT_MODEL=.*', f'DEFAULT_MODEL={selected_model}', content)
            env_file.write_text(content)
            print(f"✅ Updated .env with model: {selected_model}")
        
        return True
        
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"⚠️  Could not pull Ollama model: {e}")
        return False


def print_next_steps():
    """Print what to do next"""
    os_type = platform.system()
    
    if os_type == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        run_cmd = "python -m opensable"
    else:
        activate_cmd = "source venv/bin/activate"
        run_cmd = "python -m opensable"
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ✅ Installation Complete!                               ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

📝 Next Steps:

1. Activate virtual environment:
   {activate_cmd}

2. Edit .env file with your bot token:
   - Get token from @BotFather on Telegram
   - Set: TELEGRAM_BOT_TOKEN=your_token_here

3. Start Open-Sable:
   {run_cmd}

4. Send /start to your bot on Telegram!

📚 Documentation:
   - README.md - Feature overview
   - INSTALL.md - Detailed installation guide
   - SECURITY.md - Security features

🐛 Issues? https://github.com/IdeoaLabs/Open-Sable/issues

🎉 Enjoy your AI agent!
""")


def install_extras():
    """Ask user if they want extra features"""
    print("\n🎨 Optional Features:")
    print("   [1] Core only (minimal)")
    print("   [2] Core + Voice (speech-to-text, text-to-speech)")
    print("   [3] Core + Vision (image recognition, OCR)")
    print("   [4] All features (voice, vision, automation, monitoring)")
    
    choice = input("\n   Select option [1-4] (default: 1): ").strip() or "1"
    
    extras_map = {
        "1": "",
        "2": "[voice]",
        "3": "[vision]",
        "4": "[voice,vision,automation,database,monitoring]"
    }
    
    extras = extras_map.get(choice, "")
    
    if extras:
        print(f"\n📦 Installing extra features: {extras}...")
        venv_python = get_venv_python()
        try:
            subprocess.run(
                [str(venv_python), '-m', 'pip', 'install', '-e', f'.{extras}'],
                check=True
            )
            print("✅ Extra features installed")
        except subprocess.CalledProcessError:
            print("⚠️  Some extras failed to install (you can install later)")


def main():
    """Main installation flow"""
    print_banner()
    
    # Check requirements
    check_python_version()
    
    # Create venv
    if not create_venv():
        print("\n❌ Installation failed: Could not create virtual environment")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed at dependencies")
        sys.exit(1)
    
    # Ask for extras
    install_extras()
    
    # Setup config
    setup_env_file()
    
    # Install Ollama automatically
    ollama_installed = install_ollama()
    if ollama_installed:
        pull_ollama_model()
    else:
        print("\n⚠️  Skipping model download - install Ollama manually later")
    
    # Optional playwright
    install_playwright()
    
    # Install Venom Bot for WhatsApp
    install_venom_bot()
    
    # Done
    print_next_steps()


if __name__ == "__main__":
    main()
