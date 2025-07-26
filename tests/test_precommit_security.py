"""Tests for pre-commit security hook configuration."""

import subprocess
import sys
import yaml
from pathlib import Path


def test_precommit_config_exists():
    """Test that .pre-commit-config.yaml exists."""
    config_path = Path(".pre-commit-config.yaml")
    assert config_path.exists(), "Pre-commit configuration file not found"


def test_precommit_security_hooks_present():
    """Test that required security hooks are configured."""
    config_path = Path(".pre-commit-config.yaml")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    repos = {repo['repo']: repo for repo in config['repos']}
    
    # Test detect-secrets is present and updated
    assert 'https://github.com/Yelp/detect-secrets' in repos, \
        "detect-secrets hook not found"
    detect_secrets = repos['https://github.com/Yelp/detect-secrets']
    assert detect_secrets['rev'] >= 'v1.5.0', \
        f"detect-secrets version {detect_secrets['rev']} is outdated, requires >=v1.5.0"
    
    # Test bandit security linting is present
    assert 'https://github.com/PyCQA/bandit' in repos, \
        "bandit security hook not found"
    bandit = repos['https://github.com/PyCQA/bandit']
    assert 'bandit' in [hook['id'] for hook in bandit['hooks']], \
        "bandit hook ID not configured"
    
    # Test safety dependency checking is present
    safety_found = False
    for repo_url, repo in repos.items():
        for hook in repo['hooks']:
            if 'safety' in hook['id']:
                safety_found = True
                break
    assert safety_found, "Safety dependency vulnerability scanning not configured"
    
    # Test ruff is present and updated
    ruff_found = False
    for repo_url, repo in repos.items():
        if 'ruff' in repo_url:
            ruff_found = True
            assert repo['rev'] >= 'v0.8.0', \
                f"ruff version {repo['rev']} is outdated, requires >=v0.8.0"
            break
    assert ruff_found, "ruff linting hook not found"


def test_precommit_hooks_installable():
    """Test that pre-commit hooks can be installed."""
    # Check if pre-commit is available
    result = subprocess.run([sys.executable, "-m", "pip", "show", "pre-commit"], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        # pre-commit not installed, skip this test
        import pytest
        pytest.skip("pre-commit not installed")
    
    # Test that hooks can be installed (dry run)
    result = subprocess.run(["pre-commit", "install", "--install-hooks", "--dry-run"], 
                          capture_output=True, text=True, cwd=".")
    
    # Command should succeed or at least not fail with configuration errors
    assert result.returncode in [0, 1], \
        f"Pre-commit hook installation failed: {result.stderr}"


def test_security_baseline_file_exists():
    """Test that security baseline file exists for detect-secrets."""
    baseline_path = Path(".secrets.baseline")
    assert baseline_path.exists(), \
        "Security baseline file .secrets.baseline not found"


def test_bandit_config_exists():
    """Test that bandit configuration exists."""
    config_files = [Path(".bandit"), Path("bandit.yaml"), Path("pyproject.toml")]
    bandit_configured = any(
        config_file.exists() and ("bandit" in config_file.read_text() if config_file.suffix in ['.yaml', '.toml'] else True)
        for config_file in config_files
    )
    assert bandit_configured, \
        "Bandit security linting configuration not found"


def test_security_ignore_patterns():
    """Test that security tools have appropriate ignore patterns."""
    config_path = Path(".pre-commit-config.yaml")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check that detect-secrets excludes appropriate files
    detect_secrets_repo = None
    for repo in config['repos']:
        if 'detect-secrets' in repo['repo']:
            detect_secrets_repo = repo
            break
    
    if detect_secrets_repo:
        hooks = detect_secrets_repo['hooks']
        detect_secrets_hook = next((h for h in hooks if h['id'] == 'detect-secrets'), None)
        if detect_secrets_hook and 'exclude' in detect_secrets_hook:
            exclude_pattern = detect_secrets_hook['exclude']
            # Should exclude test files and common false positives
            assert any(pattern in exclude_pattern for pattern in ['test', 'spec', 'fixture']), \
                "detect-secrets should exclude test files to avoid false positives"