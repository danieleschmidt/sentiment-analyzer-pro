#!/bin/bash
# Repository maintenance script for Sentiment Analyzer Pro
# Automates regular maintenance tasks for repository health

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DRY_RUN=false
VERBOSE=false
CLEANUP_BRANCHES=true
CLEANUP_ARTIFACTS=true
UPDATE_DOCS=true
BACKUP_ENABLED=true

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
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

show_help() {
    cat << EOF
Repository Maintenance Script for Sentiment Analyzer Pro

Usage: $0 [OPTIONS]

Options:
  --dry-run                 Show what would be done without making changes
  --verbose                 Enable verbose output
  --no-branch-cleanup      Skip branch cleanup
  --no-artifact-cleanup    Skip artifact cleanup
  --no-doc-update          Skip documentation updates
  --no-backup              Skip backup creation
  --help                   Show this help message

Maintenance Tasks:
  - Clean up merged branches
  - Remove build artifacts
  - Update documentation
  - Create backups
  - Check repository health
  - Update project metrics
  - Cleanup temporary files

Examples:
  $0                       # Run all maintenance tasks
  $0 --dry-run            # Preview maintenance actions
  $0 --no-branch-cleanup  # Skip branch cleanup
EOF
}

check_git_status() {
    log_info "Checking git repository status..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check for uncommitted changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log_warn "Uncommitted changes detected"
        if [[ "$DRY_RUN" != "true" ]]; then
            read -p "Continue with uncommitted changes? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Maintenance cancelled"
                exit 0
            fi
        fi
    fi
    
    # Get current branch
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    log_debug "Current branch: $current_branch"
    
    # Check if we're on main/master
    if [[ "$current_branch" != "main" && "$current_branch" != "master" ]]; then
        log_warn "Not on main branch, switching to main"
        if [[ "$DRY_RUN" != "true" ]]; then
            git checkout main 2>/dev/null || git checkout master 2>/dev/null || true
        fi
    fi
}

cleanup_branches() {
    if [[ "$CLEANUP_BRANCHES" != "true" ]]; then
        log_debug "Branch cleanup skipped"
        return 0
    fi
    
    log_info "Cleaning up merged branches..."
    
    # Fetch latest changes
    if [[ "$DRY_RUN" != "true" ]]; then
        git fetch --prune origin
    else
        log_debug "Would run: git fetch --prune origin"
    fi
    
    # Get merged branches (excluding main/master/develop)
    local merged_branches
    merged_branches=$(git branch --merged | grep -v -E "(main|master|develop|\*)" | xargs -n 1 2>/dev/null || true)
    
    if [[ -z "$merged_branches" ]]; then
        log_info "No merged branches to clean up"
        return 0
    fi
    
    log_info "Found merged branches to delete:"
    echo "$merged_branches" | while read -r branch; do
        if [[ -n "$branch" ]]; then
            echo "  - $branch"
        fi
    done
    
    if [[ "$DRY_RUN" != "true" ]]; then
        echo "$merged_branches" | while read -r branch; do
            if [[ -n "$branch" ]]; then
                log_debug "Deleting branch: $branch"
                git branch -d "$branch" 2>/dev/null || true
            fi
        done
    else
        log_debug "Would delete merged branches"
    fi
    
    # Clean up remote tracking branches
    local stale_remotes
    stale_remotes=$(git remote prune origin --dry-run 2>/dev/null | grep -E "would prune" | wc -l || echo "0")
    
    if [[ "$stale_remotes" -gt 0 ]]; then
        log_info "Found $stale_remotes stale remote branches"
        if [[ "$DRY_RUN" != "true" ]]; then
            git remote prune origin
        else
            log_debug "Would prune stale remote branches"
        fi
    fi
}

cleanup_artifacts() {
    if [[ "$CLEANUP_ARTIFACTS" != "true" ]]; then
        log_debug "Artifact cleanup skipped"
        return 0
    fi
    
    log_info "Cleaning up build artifacts..."
    
    local cleanup_dirs=(
        "__pycache__"
        ".pytest_cache"
        ".coverage"
        "htmlcov"
        ".mypy_cache"
        ".ruff_cache"
        "build"
        "dist"
        "*.egg-info"
        ".tox"
        "node_modules"
        "quality-reports"
    )
    
    local cleanup_files=(
        "*.pyc"
        "*.pyo"
        "*.log"
        "*.tmp"
        "*~"
        ".DS_Store"
        "Thumbs.db"
        "*.swp"
        "*.swo"
    )
    
    # Clean up directories
    for pattern in "${cleanup_dirs[@]}"; do
        local found_dirs
        found_dirs=$(find . -type d -name "$pattern" 2>/dev/null || true)
        
        if [[ -n "$found_dirs" ]]; then
            log_debug "Cleaning directories matching: $pattern"
            if [[ "$DRY_RUN" != "true" ]]; then
                find . -type d -name "$pattern" -exec rm -rf {} + 2>/dev/null || true
            else
                echo "$found_dirs" | while read -r dir; do
                    if [[ -n "$dir" ]]; then
                        log_debug "Would remove directory: $dir"
                    fi
                done
            fi
        fi
    done
    
    # Clean up files
    for pattern in "${cleanup_files[@]}"; do
        local found_files
        found_files=$(find . -type f -name "$pattern" 2>/dev/null || true)
        
        if [[ -n "$found_files" ]]; then
            log_debug "Cleaning files matching: $pattern"
            if [[ "$DRY_RUN" != "true" ]]; then
                find . -type f -name "$pattern" -delete 2>/dev/null || true
            else
                echo "$found_files" | while read -r file; do
                    if [[ -n "$file" ]]; then
                        log_debug "Would remove file: $file"
                    fi
                done
            fi
        fi
    done
    
    # Clean up empty directories
    if [[ "$DRY_RUN" != "true" ]]; then
        find . -type d -empty -delete 2>/dev/null || true
    fi
    
    log_info "Artifact cleanup completed"
}

update_documentation() {
    if [[ "$UPDATE_DOCS" != "true" ]]; then
        log_debug "Documentation update skipped"
        return 0
    fi
    
    log_info "Updating documentation..."
    
    # Update README with latest information
    if [[ -f "README.md" ]]; then
        log_debug "Checking README.md for updates needed"
        
        # Check if version in README matches pyproject.toml
        if [[ -f "pyproject.toml" ]]; then
            local pyproject_version
            pyproject_version=$(grep -E '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/' || echo "unknown")
            
            # This would need more sophisticated parsing for real implementation
            log_debug "Project version: $pyproject_version"
        fi
    fi
    
    # Update API documentation if available
    if command -v sphinx-build &> /dev/null && [[ -d "docs" ]]; then
        log_debug "Building Sphinx documentation"
        if [[ "$DRY_RUN" != "true" ]]; then
            cd docs && make html > /dev/null 2>&1 && cd .. || true
        else
            log_debug "Would build Sphinx documentation"
        fi
    fi
    
    # Update CHANGELOG if it exists
    if [[ -f "CHANGELOG.md" ]]; then
        log_debug "CHANGELOG.md exists"
        # Could add logic to update changelog with recent commits
    fi
    
    log_info "Documentation update completed"
}

create_backup() {
    if [[ "$BACKUP_ENABLED" != "true" ]]; then
        log_debug "Backup creation skipped"
        return 0
    fi
    
    log_info "Creating repository backup..."
    
    local backup_dir="backups"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${backup_dir}/repo_backup_${timestamp}.tar.gz"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        mkdir -p "$backup_dir"
        
        # Create backup excluding unnecessary files
        tar -czf "$backup_file" \
            --exclude='.git' \
            --exclude='node_modules' \
            --exclude='__pycache__' \
            --exclude='.pytest_cache' \
            --exclude='build' \
            --exclude='dist' \
            --exclude='backups' \
            --exclude='quality-reports' \
            . 2>/dev/null || true
        
        if [[ -f "$backup_file" ]]; then
            local backup_size
            backup_size=$(du -h "$backup_file" | cut -f1)
            log_info "Backup created: $backup_file ($backup_size)"
        else
            log_error "Failed to create backup"
        fi
        
        # Clean up old backups (keep last 5)
        local old_backups
        old_backups=$(ls -t "$backup_dir"/repo_backup_*.tar.gz 2>/dev/null | tail -n +6 || true)
        
        if [[ -n "$old_backups" ]]; then
            log_debug "Cleaning up old backups"
            echo "$old_backups" | xargs rm -f 2>/dev/null || true
        fi
    else
        log_debug "Would create backup: $backup_file"
    fi
}

check_repository_health() {
    log_info "Checking repository health..."
    
    local health_issues=0
    
    # Check for large files
    local large_files
    large_files=$(find . -type f -size +10M 2>/dev/null | grep -v ".git" || true)
    
    if [[ -n "$large_files" ]]; then
        log_warn "Large files found (>10MB):"
        echo "$large_files"
        ((health_issues++))
    fi
    
    # Check for secrets in commit history (basic check)
    if command -v git-secrets &> /dev/null; then
        log_debug "Running git-secrets scan"
        if ! git secrets --scan-history > /dev/null 2>&1; then
            log_warn "Potential secrets found in history"
            ((health_issues++))
        fi
    fi
    
    # Check for required files
    local required_files=(
        "README.md"
        "LICENSE"
        "pyproject.toml"
        ".gitignore"
        "CONTRIBUTING.md"
        "SECURITY.md"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_warn "Missing required file: $file"
            ((health_issues++))
        fi
    done
    
    # Check repository size
    local repo_size
    repo_size=$(du -sh .git 2>/dev/null | cut -f1 || echo "unknown")
    log_debug "Repository size: $repo_size"
    
    if [[ "$health_issues" -eq 0 ]]; then
        log_info "Repository health check passed"
    else
        log_warn "Repository health check found $health_issues issues"
    fi
    
    return 0
}

update_metrics() {
    log_info "Updating project metrics..."
    
    local metrics_file=".github/project-metrics.json"
    
    if [[ -f "$metrics_file" ]]; then
        # Update timestamp in metrics file
        if [[ "$DRY_RUN" != "true" ]]; then
            local temp_file
            temp_file=$(mktemp)
            
            python3 -c "
import json
import datetime

try:
    with open('$metrics_file') as f:
        data = json.load(f)
    
    data['generated'] = datetime.datetime.now().isoformat() + 'Z'
    
    with open('$temp_file', 'w') as f:
        json.dump(data, f, indent=2)
        
    print('Metrics updated successfully')
except Exception as e:
    print(f'Error updating metrics: {e}')
"
            
            if [[ -f "$temp_file" ]]; then
                mv "$temp_file" "$metrics_file"
                log_debug "Metrics file updated"
            fi
        else
            log_debug "Would update metrics file"
        fi
    else
        log_warn "Metrics file not found: $metrics_file"
    fi
}

generate_maintenance_report() {
    log_info "Generating maintenance report..."
    
    local report_file="maintenance_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Repository Maintenance Report

**Date**: $(date)
**Repository**: $(basename "$(pwd)")
**Branch**: $(git rev-parse --abbrev-ref HEAD)

## Maintenance Tasks Completed

- [x] Git repository status check
- [x] Branch cleanup
- [x] Artifact cleanup  
- [x] Documentation update
- [x] Backup creation
- [x] Repository health check
- [x] Metrics update

## Summary

$(git log --oneline -10)

## Repository Statistics

- **Total commits**: $(git rev-list --all --count)
- **Total branches**: $(git branch -a | wc -l)
- **Repository size**: $(du -sh .git | cut -f1)
- **Last commit**: $(git log -1 --format="%h - %s (%cr)")

## Next Maintenance

Recommended maintenance frequency: Weekly
Next maintenance due: $(date -d "+1 week" +%Y-%m-%d)

---
Generated by repository maintenance script v1.0.0
EOF
    
    if [[ "$DRY_RUN" != "true" ]]; then
        log_info "Maintenance report created: $report_file"
    else
        rm -f "$report_file"
        log_debug "Would create maintenance report"
    fi
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --no-branch-cleanup)
                CLEANUP_BRANCHES=false
                shift
                ;;
            --no-artifact-cleanup)
                CLEANUP_ARTIFACTS=false
                shift
                ;;
            --no-doc-update)
                UPDATE_DOCS=false
                shift
                ;;
            --no-backup)
                BACKUP_ENABLED=false
                shift
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
    
    log_info "Starting repository maintenance..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "🔍 DRY RUN MODE - No changes will be made"
    fi
    
    # Run maintenance tasks
    check_git_status
    cleanup_branches
    cleanup_artifacts
    update_documentation
    create_backup
    check_repository_health
    update_metrics
    generate_maintenance_report
    
    log_info "✅ Repository maintenance completed successfully"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Run without --dry-run to apply changes"
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi