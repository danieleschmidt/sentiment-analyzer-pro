# Security Workflow Enhancements

This document outlines recommended security enhancements for the CI/CD pipeline. These changes should be implemented by updating the `.github/workflows/python-ci.yml` file.

## Current Security Measures

The current CI pipeline includes:
- Ruff linting for code quality
- Bandit security scanning for Python code
- Pytest for comprehensive testing

## Recommended Enhancements

### 1. Dependency Vulnerability Scanning

Add `safety` check to scan for known vulnerabilities in dependencies:

```yaml
- name: Install security tools
  run: pip install safety

- name: Check for security vulnerabilities
  run: safety check
```

### 2. SAST (Static Application Security Testing)

Add CodeQL for advanced static analysis:

```yaml
- name: Initialize CodeQL
  uses: github/codeql-action/init@v2
  with:
    languages: python

- name: Perform CodeQL Analysis
  uses: github/codeql-action/analyze@v2
```

### 3. License Compliance

Add license scanning:

```yaml
- name: Install license checker
  run: pip install pip-licenses

- name: Check licenses
  run: pip-licenses --format=json --output-file=licenses.json
```

### 4. SBOM Generation

Generate Software Bill of Materials:

```yaml
- name: Generate SBOM
  uses: anchore/sbom-action@v0
  with:
    path: ./
    format: spdx-json
```

### 5. Container Security (if using Docker)

Add Trivy for container vulnerability scanning:

```yaml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'sentiment-analyzer-pro:latest'
    format: 'sarif'
    output: 'trivy-results.sarif'
```

### 6. Secret Scanning

The repository already has pre-commit hooks with detect-secrets. Consider adding:

```yaml
- name: Run detect-secrets
  run: |
    pip install detect-secrets
    detect-secrets scan --all-files --exclude-files '^\.git/'
```

### 7. Security Headers Check (for web API)

```yaml
- name: Test security headers
  run: |
    python -m src.webapp &
    sleep 5
    curl -I http://localhost:5000 | grep -E "(X-Frame-Options|X-Content-Type-Options|X-XSS-Protection)"
```

## Implementation Priority

1. **High Priority**: Safety vulnerability scanning
2. **Medium Priority**: CodeQL static analysis, SBOM generation
3. **Low Priority**: License compliance, container security

## Manual Implementation Required

These enhancements require manual implementation in the GitHub workflow file as CI/CD configurations cannot be automatically modified for security reasons.

## Testing Locally

Before implementing in CI, test these tools locally:

```bash
# Install security tools
pip install safety pip-licenses detect-secrets

# Run security checks
safety check
pip-licenses
detect-secrets scan --all-files
```