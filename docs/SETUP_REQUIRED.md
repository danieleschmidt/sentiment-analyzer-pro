# Manual Setup Requirements

## Repository Configuration

The following items require manual setup by repository administrators with appropriate permissions:

### GitHub Actions Workflows
- **CI/CD Pipeline**: Create `.github/workflows/ci.yml` for automated testing
- **Security Scanning**: Create `.github/workflows/security.yml` for vulnerability detection  
- **Release Automation**: Create `.github/workflows/release.yml` for version management

See [docs/workflows/README.md](workflows/README.md) for detailed workflow specifications.

### Branch Protection
- Enable branch protection for `main` branch
- Require pull request reviews before merging
- Require status checks to pass
- Restrict direct pushes to main

### Security Settings
- Enable Dependabot alerts for vulnerability scanning
- Configure CodeQL for code security analysis
- Enable secret scanning for credential detection

### Repository Metadata
- Add repository topics: `python`, `sentiment-analysis`, `machine-learning`
- Set repository description: "Advanced sentiment analysis toolkit"
- Configure homepage URL if applicable

### External Integrations
- **Monitoring**: Configure application monitoring (optional)
- **PyPI Publishing**: Set up PyPI tokens for automated releases (optional)
- **Documentation Hosting**: Set up docs deployment (optional)

## Environment Variables

For production deployments, configure these environment variables:
- `ENVIRONMENT`: Set to `production`
- Database connection strings (if applicable)
- API keys for external services (if applicable)

## Development Setup Verification

After manual setup completion, verify with:
```bash
make setup && make check
```

This ensures all development tools and workflows function correctly.