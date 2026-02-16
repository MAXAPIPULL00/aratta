# Contributing to Aratta

Thanks for your interest in contributing. Here's how to get started.

## Development Setup

```bash
git clone https://github.com/MAXAPIPULL00/aratta.git
cd aratta
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Pull Requests

1. Fork the repo and create a feature branch from `main`.
2. Add tests for any new functionality.
3. Make sure `pytest` and `ruff check` pass.
4. Open a PR with a clear description of what changed and why.

## Reporting Issues

Open an issue on GitHub. Include:

- What you expected to happen.
- What actually happened.
- Steps to reproduce.
- Python version and OS.

## License

By contributing you agree that your contributions will be licensed under the MIT License.
