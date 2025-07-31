# Manual GitHub Workflow Setup Required

Due to GitHub App permissions, the following advanced GitHub workflows need to be manually created by a repository administrator:

## Security Audit Workflow
**File**: `.github/workflows/security-audit.yml`

```yaml
name: Security Audit

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM UTC
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  security-audit:
    runs-on: ubuntu-latest
    name: Security Audit
    
    permissions:
      contents: read
      security-events: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt', '**/pyproject.toml') }}
          
      - name: Install dependencies
        run: |
          pip install -e .[ml,web]
          pip install safety bandit semgrep
          
      - name: Run Bandit security linter
        run: |
          bandit -r src -f json -o bandit-report.json || true
          bandit -r src -f txt
          
      - name: Run Safety check for dependencies
        run: |
          safety check --json --output safety-report.json || true
          safety check
          
      - name: Run Semgrep security analysis
        uses: semgrep/semgrep-action@v1
        with:
          config: auto
          generateSarif: "true"
          
      - name: Upload SARIF to GitHub
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: semgrep.sarif
          
      - name: Archive security reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json
            semgrep.sarif
```

## Dependency Review Workflow  
**File**: `.github/workflows/dependency-review.yml`

```yaml
name: Dependency Review

on:
  pull_request:
    branches: [main]

permissions:
  contents: read
  pull-requests: write

jobs:
  dependency-review:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
        
      - name: Dependency Review
        uses: actions/dependency-review-action@v3
        with:
          fail-on-severity: moderate
          fail-on-scopes: runtime
          comment-summary-in-pr: true
          
      - name: Python Dependency Audit
        uses: pypa/gh-action-pip-audit@v1.0.8
        with:
          inputs: requirements.txt pyproject.toml
          format: json
          output: pip-audit-results.json
          
      - name: Upload audit results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: dependency-audit
          path: pip-audit-results.json
```

## OpenSSF Scorecard Workflow
**File**: `.github/workflows/scorecard.yml`

```yaml
name: OpenSSF Scorecard

on:
  branch_protection_rule:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM UTC
  push:
    branches: [main]
  workflow_dispatch:

permissions: read-all

jobs:
  analysis:
    name: Scorecard Analysis
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      id-token: write
      
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          persist-credentials: false
          
      - name: Run OpenSSF Scorecard
        uses: ossf/scorecard-action@v2.3.1
        with:
          results_file: results.sarif
          results_format: sarif
          
      - name: Upload SARIF results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: results.sarif
          
      - name: Upload Scorecard results as artifact
        uses: actions/upload-artifact@v3
        with:
          name: scorecard-results
          path: results.sarif
```

## Advanced Release Workflow
**File**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v[0-9]+.[0-9]+.[0-9]+'
      - 'v[0-9]+.[0-9]+.[0-9]+-*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release (e.g., 1.0.0)'
        required: true
        type: string

permissions:
  contents: write
  packages: write
  id-token: write

jobs:
  test:
    uses: ./.github/workflows/python-ci.yml
    
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install build dependencies
        run: |
          pip install build twine
          
      - name: Build distributions
        run: python -m build
        
      - name: Check distributions
        run: twine check dist/*
        
      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: distributions
          path: dist/
          
  security-scan:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Download build artifacts
        uses: actions/download-artifact@v3
        with:
          name: distributions
          path: dist/
          
      - name: Scan for vulnerabilities
        uses: pypa/gh-action-pip-audit@v1.0.8
        with:
          inputs: dist/*.whl
          
  release:
    needs: [test, build, security-scan]
    runs-on: ubuntu-latest
    environment: release
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: Download build artifacts
        uses: actions/download-artifact@v3
        with:
          name: distributions
          path: dist/
          
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*
          generate_release_notes: true
          
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
          
  docker:
    needs: release
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Log in to container registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=tag
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

## Setup Instructions

1. **Create each workflow file** in the `.github/workflows/` directory
2. **Configure required secrets**:
   - `PYPI_API_TOKEN` for PyPI publishing
   - Any additional tokens for security scanning services
3. **Enable branch protection rules** for enhanced security scoring
4. **Configure required environments**:
   - `release` environment with appropriate approvals

## Benefits of Manual Setup

- **Enhanced Security**: Weekly security audits and dependency reviews
- **Automated Releases**: Comprehensive build, test, and publish pipeline  
- **Compliance Monitoring**: OpenSSF Scorecard for security posture tracking
- **Quality Gates**: Security scanning before all releases

These workflows complement the existing CI/CD pipeline with enterprise-grade security and compliance automation.