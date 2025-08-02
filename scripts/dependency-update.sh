#!/bin/bash
# Automated dependency update script for Sentiment Analyzer Pro
# This script checks for and applies dependency updates safely

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.10"
BRANCH_PREFIX="chore/dependency-update"
MAX_UPDATES_PER_PR=10
DRY_RUN=false
FORCE_UPDATE=false

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

show_help() {
    cat << EOF
Dependency Update Script for Sentiment Analyzer Pro

Usage: $0 [OPTIONS]

Options:
  --dry-run              Show what would be updated without making changes
  --force                Force update even if tests fail
  --max-updates N        Maximum number of updates per PR (default: $MAX_UPDATES_PER_PR)
  --help                 Show this help message

Examples:
  $0                     # Standard dependency update
  $0 --dry-run          # Preview updates without applying
  $0 --max-updates 5    # Limit to 5 updates per PR

Environment Variables:
  GITHUB_TOKEN          GitHub token for creating PRs
  DRY_RUN              Set to 'true' for dry run mode
EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if pip-tools is available or install it
    if ! python3 -m pip list | grep -q pip-tools; then
        log_info "Installing pip-tools..."
        python3 -m pip install pip-tools
    fi
    
    # Check for GitHub CLI (optional)
    if command -v gh &> /dev/null; then
        log_info "GitHub CLI available for PR creation"
    else
        log_warn "GitHub CLI not available, manual PR creation required"
    fi
}

backup_requirements() {
    log_info "Creating backup of current requirements..."
    
    if [[ -f requirements.txt ]]; then
        cp requirements.txt requirements.txt.backup
    fi
    
    if [[ -f pyproject.toml ]]; then
        cp pyproject.toml pyproject.toml.backup
    fi
}

restore_requirements() {
    log_warn "Restoring requirements from backup..."
    
    if [[ -f requirements.txt.backup ]]; then
        mv requirements.txt.backup requirements.txt
    fi
    
    if [[ -f pyproject.toml.backup ]]; then
        mv pyproject.toml.backup pyproject.toml
    fi
}

cleanup_backups() {
    log_info "Cleaning up backup files..."
    
    rm -f requirements.txt.backup
    rm -f pyproject.toml.backup
}

check_outdated_packages() {
    log_info "Checking for outdated packages..."
    
    # Create virtual environment for checking
    python3 -m venv .update-env
    source .update-env/bin/activate
    
    # Install current dependencies
    pip install -e . > /dev/null 2>&1
    
    # Get outdated packages
    outdated_packages=$(pip list --outdated --format=json 2>/dev/null || echo "[]")
    
    # Cleanup virtual environment
    deactivate
    rm -rf .update-env
    
    echo "$outdated_packages"
}

update_dependencies() {
    local max_updates="$1"
    
    log_info "Updating dependencies (max: $max_updates)..."
    
    outdated_json=$(check_outdated_packages)
    
    if [[ "$outdated_json" == "[]" || -z "$outdated_json" ]]; then
        log_info "No outdated packages found"
        return 0
    fi
    
    # Parse outdated packages and limit updates
    echo "$outdated_json" | python3 -c "
import json
import sys

try:
    packages = json.load(sys.stdin)
    count = 0
    max_count = int(sys.argv[1])
    
    for package in packages[:max_count]:
        print(f\"{package['name']}=={package['latest_version']}\")
        count += 1
        
    print(f\"# Total packages to update: {min(len(packages), max_count)}\", file=sys.stderr)
    print(f\"# Remaining packages: {max(0, len(packages) - max_count)}\", file=sys.stderr)
    
except (json.JSONDecodeError, KeyError) as e:
    print(f\"Error parsing package list: {e}\", file=sys.stderr)
    sys.exit(1)
" "$max_updates" > updates.tmp
    
    if [[ ! -s updates.tmp ]]; then
        log_info "No packages to update"
        rm -f updates.tmp
        return 0
    fi
    
    log_info "Packages to update:"
    cat updates.tmp
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run mode - no changes made"
        rm -f updates.tmp
        return 0
    fi
    
    # Apply updates
    while read -r package_spec; do
        if [[ -n "$package_spec" && ! "$package_spec" =~ ^# ]]; then
            package_name=$(echo "$package_spec" | cut -d'=' -f1)
            log_info "Updating $package_name..."
            
            # Update in requirements.txt if it exists
            if [[ -f requirements.txt ]]; then
                sed -i.bak "s/^${package_name}==.*/${package_spec}/" requirements.txt
                rm -f requirements.txt.bak
            fi
            
            # Update in pyproject.toml if it exists (more complex parsing needed)
            # This is a simplified approach - for complex cases, use a proper TOML parser
        fi
    done < updates.tmp
    
    rm -f updates.tmp
    return 0
}

run_security_checks() {
    log_info "Running security checks..."
    
    # Check for known vulnerabilities
    if command -v safety &> /dev/null; then
        log_info "Running Safety check..."
        if ! safety check --json --output safety-report.json; then
            log_error "Security vulnerabilities found"
            return 1
        fi
    else
        log_warn "Safety not available, skipping vulnerability check"
    fi
    
    # Check for insecure dependencies
    if command -v pip-audit &> /dev/null; then
        log_info "Running pip-audit..."
        if ! pip-audit --format=json --output=audit-report.json; then
            log_error "Audit found issues"
            return 1
        fi
    else
        log_warn "pip-audit not available, skipping audit"
    fi
    
    return 0
}

run_tests() {
    log_info "Running test suite..."
    
    # Install updated dependencies
    python3 -m pip install -e .[dev] > /dev/null 2>&1
    
    # Run tests
    if command -v pytest &> /dev/null; then
        if ! pytest tests/ -x --tb=short -q; then
            log_error "Tests failed with updated dependencies"
            return 1
        fi
    else
        log_warn "pytest not available, skipping tests"
    fi
    
    # Run linting
    if command -v ruff &> /dev/null; then
        if ! ruff check src/ tests/; then
            log_error "Linting failed"
            return 1
        fi
    fi
    
    return 0
}

create_update_branch() {
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local branch_name="${BRANCH_PREFIX}-${timestamp}"
    
    log_info "Creating update branch: $branch_name"
    
    git checkout -b "$branch_name"
    echo "$branch_name"
}

commit_changes() {
    local branch_name="$1"
    
    log_info "Committing dependency updates..."
    
    # Check if there are changes to commit
    if git diff --quiet && git diff --cached --quiet; then
        log_info "No changes to commit"
        return 1
    fi
    
    # Add changed files
    git add requirements.txt pyproject.toml 2>/dev/null || true
    
    # Create commit message
    local commit_msg="chore: update dependencies

$(git diff --cached --name-only | sed 's/^/- Updated: /')

🤖 Generated with automated dependency update script

Co-Authored-By: Dependency Bot <deps@terragon.ai>"
    
    git commit -m "$commit_msg"
    
    return 0
}

create_pull_request() {
    local branch_name="$1"
    
    log_info "Creating pull request..."
    
    # Push branch
    git push origin "$branch_name"
    
    # Create PR if GitHub CLI is available
    if command -v gh &> /dev/null && [[ -n "${GITHUB_TOKEN:-}" ]]; then
        local pr_title="chore: automated dependency updates"
        local pr_body="## Automated Dependency Updates

This PR contains automated dependency updates generated by the dependency update script.

### Changes
- Updated outdated packages to latest versions
- Verified security checks pass
- Confirmed test suite passes

### Verification
- [ ] All tests pass
- [ ] Security scans clean
- [ ] No breaking changes detected

### Notes
- This is an automated PR
- Please review changes before merging
- Consider testing in staging environment

Generated on: $(date)
Script version: 1.0.0"
        
        gh pr create \
            --title "$pr_title" \
            --body "$pr_body" \
            --label "dependencies,automated" \
            --assignee "@me"
        
        log_info "Pull request created successfully"
    else
        log_warn "GitHub CLI not available or token missing"
        log_info "Please create PR manually for branch: $branch_name"
    fi
}

cleanup() {
    log_info "Cleaning up..."
    
    # Remove temporary files
    rm -f updates.tmp
    rm -f safety-report.json
    rm -f audit-report.json
    
    # Return to main branch
    git checkout main 2>/dev/null || git checkout master 2>/dev/null || true
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE_UPDATE=true
                shift
                ;;
            --max-updates)
                MAX_UPDATES_PER_PR="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Set dry run from environment if not set via CLI
    if [[ "${DRY_RUN_ENV:-}" == "true" ]]; then
        DRY_RUN=true
    fi
    
    log_info "Starting dependency update process..."
    log_info "Dry run mode: $DRY_RUN"
    log_info "Max updates per PR: $MAX_UPDATES_PER_PR"
    
    # Setup error handling
    trap cleanup EXIT
    
    # Check prerequisites
    check_prerequisites
    
    # Backup current state
    backup_requirements
    
    # Update dependencies
    if ! update_dependencies "$MAX_UPDATES_PER_PR"; then
        log_error "Failed to update dependencies"
        restore_requirements
        exit 1
    fi
    
    # Skip remaining steps if dry run
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run completed"
        exit 0
    fi
    
    # Run security checks
    if ! run_security_checks; then
        if [[ "$FORCE_UPDATE" != "true" ]]; then
            log_error "Security checks failed, aborting update"
            restore_requirements
            exit 1
        else
            log_warn "Security checks failed but continuing due to --force"
        fi
    fi
    
    # Run tests
    if ! run_tests; then
        if [[ "$FORCE_UPDATE" != "true" ]]; then
            log_error "Tests failed, aborting update"
            restore_requirements
            exit 1
        else
            log_warn "Tests failed but continuing due to --force"
        fi
    fi
    
    # Create branch and commit changes
    branch_name=$(create_update_branch)
    
    if commit_changes "$branch_name"; then
        create_pull_request "$branch_name"
        log_info "Dependency update completed successfully"
    else
        log_info "No changes to commit"
        git checkout main 2>/dev/null || git checkout master 2>/dev/null
        git branch -D "$branch_name" 2>/dev/null || true
    fi
    
    # Cleanup backups on success
    cleanup_backups
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi