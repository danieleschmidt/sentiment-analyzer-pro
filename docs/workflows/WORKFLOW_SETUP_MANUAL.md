# Manual Workflow Setup Guide

This guide provides step-by-step instructions for manually setting up GitHub Actions workflows for the Sentiment Analyzer Pro repository.

## Prerequisites

### Required Permissions
- Repository administrator access
- Ability to create and manage GitHub Actions workflows
- Access to repository settings and security features

### Required Secrets
Set up the following secrets in your repository settings (`Settings > Secrets and variables > Actions`):

| Secret Name | Description | Required For |
|-------------|-------------|--------------|
| `PYPI_API_TOKEN` | PyPI API token for package publishing | Release workflow |
| `CODECOV_TOKEN` | Codecov token for coverage reporting | CI workflow |
| `SLACK_WEBHOOK` | Slack webhook for notifications | All workflows |

## Step 1: Create Workflow Files

### 1.1 Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### 1.2 Copy Workflow Templates
Copy the example workflows from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Copy CI/CD workflow
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml

# Copy security scanning workflow
cp docs/workflows/examples/security.yml .github/workflows/security.yml

# Copy release workflow
cp docs/workflows/examples/release.yml .github/workflows/release.yml
```

### 1.3 Customize Workflows
Edit each workflow file to match your specific requirements:

#### CI Workflow Customizations
- Update Python versions in the matrix
- Modify test commands if using different test runners
- Adjust coverage thresholds
- Configure notification preferences

#### Security Workflow Customizations
- Set appropriate scan schedules
- Configure security tool preferences
- Adjust vulnerability thresholds
- Set up compliance requirements

#### Release Workflow Customizations
- Configure registry settings
- Set up deployment targets
- Customize release notes generation
- Configure notification channels

## Step 2: Repository Settings Configuration

### 2.1 Branch Protection Rules

Navigate to `Settings > Branches` and create protection rules for your main branch:

#### Required Settings:
- **Require pull request reviews before merging**
  - Number of required reviewers: 2
  - Dismiss stale reviews when new commits are pushed: ✓
  - Require review from code owners: ✓

- **Require status checks to pass before merging**
  - Require branches to be up to date before merging: ✓
  - Required status checks:
    - `Test Suite (3.9)`
    - `Test Suite (3.10)`
    - `Test Suite (3.11)`
    - `Integration Tests`
    - `Docker Build`
    - `Security Scanning`

- **Require conversation resolution before merging**: ✓
- **Restrict pushes that create files**: ✓
- **Do not allow bypassing the above settings**: ✓

### 2.2 Security Settings

Navigate to `Settings > Security` and configure:

#### Code Scanning
- **Enable CodeQL analysis**: ✓
- **Enable dependency scanning**: ✓
- **Enable secret scanning**: ✓
- **Enable secret scanning push protection**: ✓

#### Security Advisories
- **Enable private vulnerability reporting**: ✓
- **Enable automatic security updates**: ✓

### 2.3 Actions Settings

Navigate to `Settings > Actions > General`:

#### Actions Permissions
- **Allow all actions and reusable workflows**: ✓
- **Allow actions created by GitHub and verified creators**: ✓

#### Workflow Permissions
- **Read and write permissions**: ✓
- **Allow GitHub Actions to create and approve pull requests**: ✓

#### Fork Pull Request Workflows
- **Require approval for all outside collaborators**: ✓

## Step 3: Environment Configuration

### 3.1 Create Environments

Navigate to `Settings > Environments` and create:

#### Staging Environment
- **Environment name**: `staging`
- **Deployment branches**: `develop`
- **Environment secrets**: Add staging-specific secrets
- **Required reviewers**: Add staging deployment approvers

#### Production Environment
- **Environment name**: `production`
- **Deployment branches**: `main`
- **Environment secrets**: Add production-specific secrets
- **Required reviewers**: Add production deployment approvers
- **Wait timer**: 10 minutes (optional)

### 3.2 Environment Secrets

Add environment-specific secrets:

#### Staging Secrets
```
STAGING_DATABASE_URL
STAGING_REDIS_URL
STAGING_API_KEY
```

#### Production Secrets
```
PRODUCTION_DATABASE_URL
PRODUCTION_REDIS_URL
PRODUCTION_API_KEY
```

## Step 4: Webhook Configuration

### 4.1 Slack Notifications (Optional)

If using Slack notifications:

1. Create a Slack app in your workspace
2. Add incoming webhook to your app
3. Copy webhook URL to `SLACK_WEBHOOK` secret
4. Test webhook connection

### 4.2 External Services (Optional)

Configure webhooks for:
- Codecov for coverage reporting
- Sentry for error monitoring
- DataDog for performance monitoring

## Step 5: Testing and Validation

### 5.1 Test CI Workflow

1. Create a test branch:
   ```bash
   git checkout -b test-ci-setup
   echo "# Test" >> TEST.md
   git add TEST.md
   git commit -m "test: CI workflow setup"
   git push origin test-ci-setup
   ```

2. Create a pull request to trigger CI workflow
3. Verify all checks pass
4. Check that status checks appear in the PR

### 5.2 Test Security Workflow

1. Trigger security workflow manually:
   - Go to `Actions` tab
   - Select `Security Scanning` workflow
   - Click `Run workflow`

2. Verify all security scans complete
3. Check that results appear in Security tab

### 5.3 Test Release Workflow

1. Create a test release:
   ```bash
   git tag v0.1.0-test
   git push origin v0.1.0-test
   ```

2. Verify release workflow triggers
3. Check that release is created
4. Delete test release and tag

## Step 6: Monitoring and Maintenance

### 6.1 Regular Monitoring

Monitor the following regularly:
- Workflow success rates
- Build times and performance
- Security scan results
- Dependency updates

### 6.2 Workflow Updates

Keep workflows updated by:
- Reviewing action versions quarterly
- Updating security configurations
- Monitoring GitHub Actions changelog
- Testing workflow changes in feature branches

### 6.3 Performance Optimization

Optimize workflows by:
- Using caching for dependencies
- Parallelizing independent jobs
- Optimizing Docker builds
- Reducing test execution time

## Troubleshooting

### Common Issues

#### Workflow Not Triggering
**Symptoms**: Workflows don't run on push/PR
**Solutions**:
- Check branch protection settings
- Verify workflow file syntax
- Ensure correct trigger events
- Check repository permissions

#### Test Failures
**Symptoms**: CI tests fail unexpectedly
**Solutions**:
- Check test environment setup
- Verify dependency versions
- Review test data and fixtures
- Check for flaky tests

#### Security Scan Failures
**Symptoms**: Security workflows fail or find issues
**Solutions**:
- Review vulnerability reports
- Update dependencies
- Fix security issues in code
- Adjust scan configurations

#### Deployment Failures
**Symptoms**: Release or deployment workflows fail
**Solutions**:
- Check environment secrets
- Verify deployment permissions
- Review deployment logs
- Test deployment process locally

### Getting Help

If you encounter issues:

1. **Check GitHub Actions documentation**: https://docs.github.com/en/actions
2. **Review workflow logs**: Available in the Actions tab
3. **Search GitHub Community**: https://github.community/
4. **Contact repository maintainers**: Create an issue with workflow questions

## Security Considerations

### Secrets Management
- Never commit secrets to repository
- Use environment-specific secrets
- Rotate secrets regularly
- Monitor secret usage

### Permissions
- Use least-privilege principle
- Regularly review repository access
- Monitor workflow permissions
- Audit security configurations

### Compliance
- Ensure workflows meet compliance requirements
- Document security procedures
- Regular security reviews
- Keep audit trails

## Best Practices

### Workflow Design
- Keep workflows simple and focused
- Use reusable actions when possible
- Document workflow purposes
- Test workflows thoroughly

### Security
- Regular security scans
- Dependency updates
- Secret rotation
- Access reviews

### Performance
- Cache dependencies
- Parallelize jobs
- Optimize build times
- Monitor resource usage

### Maintenance
- Regular updates
- Documentation updates
- Performance monitoring
- Continuous improvement

---

This manual setup ensures comprehensive CI/CD and security workflows for the Sentiment Analyzer Pro project. Regular maintenance and monitoring will keep the workflows effective and secure.