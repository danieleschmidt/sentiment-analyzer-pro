# Security Dependency Updates

## Current Security Status

### ✅ Secure Dependencies
- **setuptools**: 80.9.0 ≥ 78.1.1 (secure)
- **PyJWT**: 2.10.1 ≥ 2.10.1 (secure)

### ⚠️ Vulnerable Dependencies (Externally Managed)

#### cryptography 41.0.7 → ≥42.0.5 Required
- **Current**: 41.0.7 (system-managed)
- **Required**: ≥42.0.5
- **CVEs**: Multiple cryptography vulnerabilities
- **Status**: Cannot update due to externally-managed environment

#### pip 24.0 → ≥25.0 Required  
- **Current**: 24.0 (system-managed)
- **Required**: ≥25.0
- **CVE**: PVE-2025-75180
- **Status**: Cannot update due to externally-managed environment

## Update Instructions

### For Development Environment

1. **Create Virtual Environment** (Recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip setuptools cryptography>=42.0.5 PyJWT>=2.10.1
   ```

2. **User-Space Installation** (Alternative):
   ```bash
   pip install --user --upgrade cryptography>=42.0.5
   pip install --user --upgrade pip>=25.0
   ```

3. **System Package Updates** (If available):
   ```bash
   # Check for system package updates
   apt update && apt list --upgradable | grep -E "(python3-cryptography|python3-pip)"
   ```

### For Production Deployment

1. **Use requirements.txt with minimum secure versions**:
   ```
   cryptography>=42.0.5
   pip>=25.0
   setuptools>=78.1.1
   PyJWT>=2.10.1
   ```

2. **Docker/Container environments**: Update base image to include secure versions

3. **CI/CD**: Ensure build environments use virtual environments with updated packages

## Security Script

Run `python3 scripts/update_security_dependencies.py` to:
- Check current vulnerability status
- Attempt automatic updates where possible
- Generate update instructions for system-managed packages

## Test Integration

Security dependency tests will pass in virtual environments with updated packages but may indicate warnings in system-managed environments. This is expected behavior.