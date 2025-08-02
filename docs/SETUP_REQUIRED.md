# Manual Setup Requirements

This document outlines the manual setup steps required to complete the SDLC implementation for the Sentiment Analyzer Pro repository due to GitHub App permission limitations.

## Overview

The Terragon SDLC implementation has successfully created comprehensive templates, documentation, and automation scripts. However, certain repository configurations require manual setup by repository administrators due to GitHub App permission restrictions.

## Required Manual Actions

### 1. GitHub Actions Workflows ⚠️ **CRITICAL**

#### Action Required
Copy workflow templates from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml
cp docs/workflows/examples/security.yml .github/workflows/security.yml
cp docs/workflows/examples/release.yml .github/workflows/release.yml
```

#### Files to Create
- `.github/workflows/ci.yml` - Continuous integration pipeline
- `.github/workflows/security.yml` - Security scanning automation  
- `.github/workflows/release.yml` - Release automation

#### Documentation
Complete setup instructions: [docs/workflows/WORKFLOW_SETUP_MANUAL.md](workflows/WORKFLOW_SETUP_MANUAL.md)

### 2. Repository Settings Configuration

#### Branch Protection Rules
Navigate to **Settings > Branches** and configure:

**For `main` branch:**
- ✅ Require pull request reviews (2 reviewers)
- ✅ Dismiss stale reviews when new commits are pushed
- ✅ Require review from code owners
- ✅ Require status checks to pass before merging
  - `Test Suite (3.9)`
  - `Test Suite (3.10)` 
  - `Test Suite (3.11)`
  - `Integration Tests`
  - `Docker Build`
  - `Security Scanning`
- ✅ Require conversation resolution before merging
- ✅ Restrict pushes that create files
- ✅ Do not allow bypassing settings

#### Security Settings  
Navigate to **Settings > Security**:
- ✅ Enable CodeQL analysis
- ✅ Enable dependency scanning
- ✅ Enable secret scanning
- ✅ Enable secret scanning push protection
- ✅ Enable private vulnerability reporting
- ✅ Enable automatic security updates

### 3. Environment Configuration

#### Create Environments
Navigate to **Settings > Environments**:

**Staging Environment:**
- Environment name: `staging`
- Deployment branches: `develop`
- Required reviewers: Add staging deployment approvers
- Environment secrets: Add staging-specific secrets

**Production Environment:**
- Environment name: `production`
- Deployment branches: `main`
- Required reviewers: Add production deployment approvers
- Wait timer: 10 minutes
- Environment secrets: Add production-specific secrets

### 4. Secrets Configuration

#### Repository Secrets
Navigate to **Settings > Secrets and variables > Actions**:

| Secret Name | Description | Usage |
|-------------|-------------|--------|
| `PYPI_API_TOKEN` | PyPI API token for package publishing | Release workflow |
| `CODECOV_TOKEN` | Codecov token for coverage reporting | CI workflow |
| `DOCKER_REGISTRY_TOKEN` | Container registry access token | Docker builds |

#### Environment Secrets

**Staging Secrets:**
```
STAGING_DATABASE_URL
STAGING_REDIS_URL
STAGING_API_KEY
```

**Production Secrets:**
```
PRODUCTION_DATABASE_URL
PRODUCTION_REDIS_URL  
PRODUCTION_API_KEY
```

### 5. External Service Integration

#### Codecov Integration
1. Sign up at [codecov.io](https://codecov.io)
2. Connect your repository
3. Copy token to `CODECOV_TOKEN` secret
4. Verify coverage reporting in PRs

#### Container Registry Setup
1. Configure GitHub Container Registry or Docker Hub
2. Generate access token
3. Add token to `DOCKER_REGISTRY_TOKEN` secret
4. Update registry URLs in workflows

### 6. Repository Topics and Description

#### Update Repository Settings
Navigate to **Settings > General**:
- **Description**: "Advanced sentiment analysis toolkit with transformer models, comprehensive testing, and production-ready SDLC"
- **Website**: Add documentation or demo URL
- **Topics**: `sentiment-analysis`, `machine-learning`, `python`, `transformer`, `nlp`, `flask`, `docker`, `ci-cd`

### 7. Team and Access Management

#### Collaborator Access
Navigate to **Settings > Manage access**:
- Add team members with appropriate roles
- Configure team-based permissions
- Set up code review assignments

#### CODEOWNERS Verification
Verify `.github/CODEOWNERS` file has correct usernames:
```
# Update with actual GitHub usernames
* @your-username @team-lead
/.github/ @your-username @team-lead
```

## Verification Steps

### 1. Test CI/CD Pipeline
```bash
# Create test branch
git checkout -b test-ci-setup
echo "# Test CI" >> TEST.md
git add TEST.md
git commit -m "test: verify CI pipeline"
git push origin test-ci-setup

# Create PR and verify:
# - All status checks run
# - Security scans complete
# - Coverage reports generate
# - Docker builds succeed
```

### 2. Test Security Features
```bash
# Trigger security workflow manually
# Verify all scans complete successfully
# Check Security tab for results
```

### 3. Test Release Process
```bash
# Create test release tag
git tag v0.1.0-test
git push origin v0.1.0-test

# Verify:
# - Release workflow triggers
# - Docker images build and push
# - GitHub release created
# - Artifacts uploaded
```

## Automation Scripts Usage

### Quality Checks
```bash
# Run comprehensive quality analysis
./scripts/quality-check.sh

# Generate quality report
./scripts/quality-check.sh --verbose
```

### Dependency Management
```bash
# Check for dependency updates
./scripts/dependency-update.sh --dry-run

# Apply dependency updates
./scripts/dependency-update.sh
```

### Repository Maintenance
```bash
# Preview maintenance tasks
./scripts/repo-maintenance.sh --dry-run

# Run maintenance
./scripts/repo-maintenance.sh
```

### Docker Build Automation
```bash
# Build production image
./scripts/build.sh

# Build and push with security scan
./scripts/build.sh --push --scan
```

## Success Criteria

✅ **Implementation Complete When:**
- [ ] All workflows are created and running
- [ ] Branch protection rules are configured
- [ ] Security features are enabled
- [ ] Environments are set up with secrets
- [ ] Team access is configured
- [ ] CI/CD pipeline is tested and working
- [ ] Security scanning is operational
- [ ] Monitoring stack is deployed
- [ ] Documentation is accessible
- [ ] Automation scripts are functional

## Implementation Summary

The Terragon SDLC implementation provides:

### ✅ Completed Automatically
- **Project Foundation**: Community files, documentation, ADR templates
- **Development Environment**: DevContainer, VSCode config, pre-commit hooks
- **Testing Infrastructure**: Comprehensive test structure, fixtures, performance tests
- **Build & Containerization**: Multi-stage Dockerfiles, docker-compose configurations
- **Monitoring & Observability**: Full monitoring stack, health checks, runbooks
- **Workflow Templates**: Complete CI/CD, security, and release workflows
- **Metrics & Automation**: Quality checks, dependency management, maintenance scripts

### ⚠️ Requires Manual Setup
- GitHub Actions workflow creation
- Repository settings configuration
- Security feature enablement
- Environment and secrets setup
- Team and access management

This comprehensive SDLC implementation transforms your repository into a production-ready, enterprise-grade development environment with industry best practices for security, quality, and operational excellence.

---

**Next Steps**: Follow the manual setup instructions above to complete the SDLC implementation. Start with the critical GitHub Actions workflows, then proceed through the other configuration steps.