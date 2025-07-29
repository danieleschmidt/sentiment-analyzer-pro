# GitHub Workflows Manual Setup Required

Due to GitHub App permissions, the following workflow enhancements need to be manually implemented by the repository owner.

## 1. Enhanced CI Pipeline

Replace the existing `.github/workflows/python-ci.yml` with this enhanced version:

```yaml
name: CI

on: 
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

permissions:
  contents: read
  security-events: write

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ hashFiles('**/pyproject.toml') }}
          restore-keys: |
            ${{ runner.os }}-pip-${{ matrix.python-version }}-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[ml,web,dev]
          pip install ruff bandit safety

      - name: Lint with ruff
        run: ruff check src tests --output-format=github

      - name: Security scan with bandit
        run: bandit -r src -f json -o bandit-report.json

      - name: Dependency vulnerability scan
        run: safety check --json --output safety-report.json

      - name: Run tests with coverage
        run: pytest -v --cov=src --cov-report=xml --cov-report=term-missing

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

      - name: Upload security reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-reports-${{ matrix.python-version }}
          path: |
            bandit-report.json
            safety-report.json

  build-test:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Check package
        run: twine check dist/*

      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: dist
          path: dist/

  container-test:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t sentiment-analyzer-pro:test .

      - name: Test container
        run: |
          docker run --rm -d -p 8080:8080 --name test-container sentiment-analyzer-pro:test
          sleep 10
          curl -f http://localhost:8080/health || exit 1
          docker stop test-container
```

## 2. Comprehensive Security Scanning

Create a new file `.github/workflows/security-scan.yml`:

```yaml
name: Security Scanning

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run daily at 2 AM UTC for continuous monitoring
    - cron: '0 2 * * *'

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install safety pip-audit cyclonedx-bom

      - name: Run Safety scan
        run: safety check --json --output safety-report.json || true

      - name: Run pip-audit scan
        run: pip-audit --format=json --output=pip-audit-report.json || true

      - name: Generate SBOM
        run: |
          pip install -e .
          cyclonedx-py -o sbom.json

      - name: Upload SBOM as artifact
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: sbom.json

      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            safety-report.json
            pip-audit-report.json

  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t sentiment-analyzer-pro:scan .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'sentiment-analyzer-pro:scan'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

  secrets-scan:
    name: Secrets Detection
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run TruffleHog
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD
          extra_args: --debug --only-verified

  static-analysis:
    name: Static Analysis Security Testing
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install bandit[toml] semgrep

      - name: Run Bandit security linter
        run: |
          bandit -r src -f json -o bandit-report.json
          bandit -r src -f sarif -o bandit-results.sarif

      - name: Run Semgrep security analysis
        run: |
          semgrep --config=auto --json --output=semgrep-report.json src/
          semgrep --config=auto --sarif --output=semgrep-results.sarif src/

      - name: Upload SARIF results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: |
            bandit-results.sarif
            semgrep-results.sarif

      - name: Upload security analysis reports
        uses: actions/upload-artifact@v3
        with:
          name: static-analysis-reports
          path: |
            bandit-report.json
            semgrep-report.json
```

## Implementation Steps

1. **Replace CI Workflow**: Update `.github/workflows/python-ci.yml` with the enhanced version above
2. **Add Security Workflow**: Create `.github/workflows/security-scan.yml` with the security scanning pipeline
3. **Enable Dependabot**: The `.github/dependabot.yml` file has been created and will be active once pushed
4. **Configure Secrets**: Add any required secrets (like `CODECOV_TOKEN`) to your repository settings

## Benefits of Manual Implementation

- **Multi-version Testing**: Tests across Python 3.9, 3.10, and 3.11
- **Enhanced Security**: Comprehensive security scanning with SARIF integration
- **SBOM Generation**: Automated software bill of materials
- **Container Security**: Trivy vulnerability scanning
- **Dependency Monitoring**: Automated vulnerability detection
- **Coverage Reporting**: Codecov integration for coverage tracking

## Security Integration

After implementing these workflows, you'll have:
- Automated security scanning on every PR and push
- Daily security scans for continuous monitoring
- SARIF integration with GitHub Security tab
- Comprehensive dependency vulnerability tracking
- Container security assessment

These enhancements will elevate your repository's SDLC maturity to enterprise-level security and operational excellence.