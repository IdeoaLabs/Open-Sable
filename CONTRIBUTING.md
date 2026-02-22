# Contributing to Open-Sable

Thank you for your interest in contributing to Open-Sable! 🎉

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/Open-Sable.git
   cd Open-Sable
   ```
3. **Set up** the development environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/my-awesome-feature
   ```

## Development Workflow

### Running Tests

```bash
pytest tests/ -v --tb=short
```

### Linting & Formatting

We use **ruff** for linting and **black** for formatting:

```bash
ruff check .          # lint
black .               # format
black --check .       # verify formatting
```

### Code Style

- Line length: 100 characters
- Python 3.11+ features are welcome
- Type hints are encouraged
- Docstrings for all public functions and classes

## Pull Request Process

1. Ensure all tests pass (`pytest tests/`)
2. Ensure linting is clean (`ruff check .`)
3. Ensure code is formatted (`black --check .`)
4. Update documentation if needed
5. Write a clear PR description

## What to Contribute

- 🐛 **Bug fixes** — always welcome
- ✨ **New skills** — add them under `opensable/skills/`
- 🔌 **New interfaces** — add them under `opensable/interfaces/`
- 📖 **Documentation** — improvements to docs, examples, README
- 🧪 **Tests** — more coverage is always appreciated
- 🌍 **Translations** — help make Open-Sable accessible

## Reporting Issues

- Use the [GitHub Issues](https://github.com/IdeoaLabs/Open-Sable/issues) tracker
- Include steps to reproduce, expected vs actual behavior
- Include Python version and OS information

## Code of Conduct

Be respectful, constructive, and inclusive. We're all here to build something great together.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
