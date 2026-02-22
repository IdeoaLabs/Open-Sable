# Installing Open-Sable

## Quick Install (Linux / macOS)

```bash
git clone https://github.com/IdeoaLabs/Open-Sable.git
cd Open-Sable
./quickstart.sh
```

## Manual Install

### Prerequisites

- **Python 3.11+**
- **Ollama** (for local LLM): https://ollama.ai

### Steps

```bash
# 1. Clone
git clone https://github.com/IdeoaLabs/Open-Sable.git
cd Open-Sable

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Set up environment
cp .env.example .env
# Edit .env with your API keys / tokens

# 5. Create required directories
mkdir -p data logs config

# 6. Run
python -m opensable
```

### Optional Extras

Install additional features as needed:

```bash
pip install -e ".[telegram]"     # Telegram bot + userbot
pip install -e ".[discord]"      # Discord bot
pip install -e ".[slack]"        # Slack integration
pip install -e ".[web]"          # Web dashboard (FastAPI)
pip install -e ".[voice]"        # Voice input/output
pip install -e ".[vision]"       # Image analysis
pip install -e ".[automation]"   # Browser automation
pip install -e ".[database]"     # Database connectors
pip install -e ".[monitoring]"   # Prometheus metrics
pip install -e ".[dev]"          # Development tools (pytest, black, ruff)
```

Or install the `core` bundle for the most common extras:

```bash
pip install -e ".[core]"         # Telegram + browser automation
```

## Docker Install

```bash
docker-compose up -d
```

See the [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml) for details.

## Verify Installation

```bash
python -m opensable --help
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Activate your venv: `source venv/bin/activate` |
| Python version error | Upgrade to Python 3.11+ |
| Ollama not found | Install from https://ollama.ai |
| ChromaDB errors | `pip install --upgrade chromadb` |

For more help, open an [issue](https://github.com/IdeoaLabs/Open-Sable/issues).
