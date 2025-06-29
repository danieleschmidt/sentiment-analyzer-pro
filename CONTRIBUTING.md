# Contributing

Thank you for your interest in contributing to Sentiment Analyzer Pro!

## Development Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full test suite before committing:
   ```bash
   pytest -q
   ```
3. Install the git hooks with `pre-commit install` to automatically run linters
   and scan for secrets. You can manually trigger all hooks with:
   ```bash
   pre-commit run --all-files
   ```

## Pull Requests
- Keep changes focused and include tests for new functionality.
- Ensure `pytest` passes and update documentation when necessary.
