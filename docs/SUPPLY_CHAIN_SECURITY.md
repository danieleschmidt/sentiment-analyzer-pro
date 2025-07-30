# Supply Chain Security

This document outlines the supply chain security measures implemented to protect against dependency vulnerabilities, malicious packages, and other supply chain attacks.

## SBOM (Software Bill of Materials)

### Automated SBOM Generation

The project automatically generates Software Bill of Materials for transparency and security auditing:

```bash
# Generate SBOM in SPDX format
pip install cyclonedx-bom
cyclonedx-py -o sbom.json
```

### SBOM Contents
- **Direct dependencies**: All packages listed in requirements.txt
- **Transitive dependencies**: Complete dependency tree
- **Versions and hashes**: Exact versions with cryptographic hashes
- **Licenses**: License information for compliance
- **Vulnerabilities**: Known security issues

## Dependency Security

### 1. Dependency Pinning
All dependencies are pinned to specific versions in `requirements.txt`:

```txt
# Core dependencies with exact versions
pandas==2.1.4
scikit-learn==1.3.2
numpy==1.24.4
nltk==3.8.1

# Security-critical dependencies
cryptography>=42.0.5  # Latest security patches
pyjwt>=2.10.1         # CVE fixes
```

### 2. Vulnerability Scanning

Multiple tools scan for known vulnerabilities:

**GitHub Dependabot**:
- Automated vulnerability alerts
- Automatic security update PRs
- Database: GitHub Advisory Database

**Safety Check**:
```bash
pip install safety
safety check --json > safety-report.json
```

**Bandit (Static Analysis)**:
```bash
bandit -r src/ -f json -o bandit-report.json
```

**Trivy (Container Scanning)**:
```bash
trivy fs --format json --output trivy-report.json .
```

### 3. License Compliance
```bash
# Check license compatibility
pip install pip-licenses
pip-licenses --format=json --output-file=licenses.json

# Verify OSS license compliance
python scripts/check_licenses.py
```

## Package Integrity

### 1. Hash Verification
All packages are verified against known good hashes:

```bash
# Generate requirements with hashes
pip-compile --generate-hashes requirements.in

# Install with hash verification
pip install --require-hashes -r requirements.txt
```

### 2. Signature Verification
Where available, verify package signatures:

```bash
# Verify PyPI signatures (when available)
pip install sigstore
python -m sigstore verify --cert identity.crt --signature signature.sig package.whl
```

### 3. Source Verification
Critical dependencies are built from source when possible:

```dockerfile
# Build from source for critical packages
RUN pip install --no-binary=cryptography cryptography
```

## Repository Security

### 1. Branch Protection
- Require signed commits
- Require status checks
- Restrict force pushes
- Require administrator review

### 2. Secrets Management
```bash
# Scan for accidentally committed secrets
pre-commit run detect-secrets --all-files

# Use environment variables for secrets
export API_KEY=$(cat /run/secrets/api_key)
```

### 3. Code Signing
```bash
# Sign releases with GPG
gpg --armor --detach-sign dist/package.tar.gz

# Verify signatures
gpg --verify dist/package.tar.gz.asc dist/package.tar.gz
```

## Container Security

### 1. Base Image Security
```dockerfile
# Use minimal, regularly updated base images
FROM python:3.9-slim-bullseye

# Update system packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Run as non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
```

### 2. Multi-stage Builds
```dockerfile
# Build stage - includes build tools
FROM python:3.9 as builder
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Runtime stage - minimal dependencies
FROM python:3.9-slim
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*
```

### 3. Container Scanning
```bash
# Scan container images
trivy image sentiment-analyzer:latest

# Continuous scanning in CI/CD
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image --exit-code 1 sentiment-analyzer:latest
```

## CI/CD Security

### 1. Secure Workflows
```yaml
# .github/workflows/security-scan.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
    
    steps:
      - uses: actions/checkout@v4
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

### 2. Artifact Attestation
```yaml
- name: Generate SLSA Provenance
  uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.7.0
  with:
    base64-subjects: "${{ steps.hash.outputs.hashes }}"
    provenance-name: "provenance.intoto.jsonl"
```

### 3. Environment Isolation
- Use separate environments for different stages
- Limit access to production secrets
- Monitor deployment activities

## Monitoring and Response

### 1. Vulnerability Monitoring
```python
# Automated vulnerability checking
import requests
import json

def check_vulnerabilities():
    # Query security databases
    response = requests.get("https://api.osv.dev/v1/query", json={
        "package": {"name": "package-name", "ecosystem": "PyPI"},
        "version": "1.2.3"
    })
    
    vulnerabilities = response.json()
    if vulnerabilities.get('vulns'):
        # Alert on new vulnerabilities
        send_security_alert(vulnerabilities)
```

### 2. Incident Response
1. **Detection**: Automated alerts for new vulnerabilities
2. **Assessment**: Evaluate impact and exploitability
3. **Response**: Update dependencies or implement mitigations
4. **Communication**: Notify stakeholders and users
5. **Recovery**: Deploy fixes and verify resolution

### 3. Security Metrics
Track security posture with metrics:
- Time to patch critical vulnerabilities
- Percentage of dependencies with known vulnerabilities
- SBOM generation coverage
- Security scan pass rate

## Compliance and Auditing

### 1. Regulatory Compliance
- **GDPR**: Data protection for EU users
- **SOC 2**: Security controls for service organizations
- **ISO 27001**: Information security management
- **NIST Framework**: Cybersecurity framework compliance

### 2. Audit Trail
```bash
# Generate audit logs
python scripts/generate_audit_report.py --from 2024-01-01 --to 2024-12-31

# Security event logging
tail -f /var/log/security.log | grep "sentiment-analyzer"
```

### 3. Third-party Audits
- Annual security audits by external firms
- Penetration testing of deployed applications
- Code review by security experts
- Compliance certification maintenance

## Emergency Procedures

### 1. Critical Vulnerability Response
```bash
#!/bin/bash
# emergency-patch.sh - Rapid response to critical vulnerabilities

# 1. Assess vulnerability
echo "Assessing vulnerability impact..."
python scripts/vulnerability_assessment.py $VULN_ID

# 2. Update dependencies
pip install --upgrade $VULNERABLE_PACKAGE

# 3. Run security tests
pytest tests/security/ -v

# 4. Generate new SBOM
cyclonedx-py -o sbom-patched.json

# 5. Deploy emergency fix
docker build -t sentiment-analyzer:emergency-patch .
kubectl set image deployment/sentiment-api app=sentiment-analyzer:emergency-patch
```

### 2. Incident Communication
Template for security incident communication:
```markdown
# Security Incident Report

**Date**: YYYY-MM-DD
**Severity**: Critical/High/Medium/Low
**Status**: Investigating/Resolved/Monitoring

## Summary
Brief description of the security incident.

## Impact
- Affected systems and users
- Data exposure risk
- Service availability impact

## Response Actions
1. Immediate containment measures
2. Investigation steps taken
3. Remediation activities

## Timeline
- Detection: YYYY-MM-DD HH:MM UTC
- Containment: YYYY-MM-DD HH:MM UTC
- Resolution: YYYY-MM-DD HH:MM UTC

## Next Steps
- Additional monitoring
- Process improvements
- Communication to stakeholders
```

## Tools and Resources

### Security Tools
- **Safety**: Python dependency vulnerability scanner
- **Bandit**: Python security linter
- **Trivy**: Comprehensive vulnerability scanner
- **Syft**: SBOM generation tool
- **Grype**: Vulnerability scanner for container images

### Security Databases
- **GitHub Advisory Database**: Comprehensive vulnerability data
- **OSV.dev**: Open Source Vulnerabilities database
- **PyUp.io**: Python package security monitoring
- **Snyk**: Commercial vulnerability database

### Best Practices References
- [SLSA Framework](https://slsa.dev/): Supply-chain Levels for Software Artifacts
- [NIST SSDF](https://csrc.nist.gov/Projects/ssdf): Secure Software Development Framework
- [OpenSSF](https://openssf.org/): Open Source Security Foundation guidelines
- [OWASP SCVS](https://owasp.org/www-project-software-component-verification-standard/): Software Component Verification Standard

For immediate security concerns, contact: security@company.com