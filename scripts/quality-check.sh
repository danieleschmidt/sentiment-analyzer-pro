#!/bin/bash
# Quality check and monitoring script for Sentiment Analyzer Pro
# Provides comprehensive quality metrics and automated monitoring

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.10"
OUTPUT_DIR="quality-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VERBOSE=false
GENERATE_REPORT=true

# Thresholds
COVERAGE_THRESHOLD=80
COMPLEXITY_THRESHOLD=10
DUPLICATION_THRESHOLD=5
SECURITY_THRESHOLD=0

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
Quality Check Script for Sentiment Analyzer Pro

Usage: $0 [OPTIONS]

Options:
  --verbose              Enable verbose output
  --no-report           Skip HTML report generation
  --output-dir DIR      Output directory for reports (default: $OUTPUT_DIR)
  --coverage-threshold N Coverage threshold percentage (default: $COVERAGE_THRESHOLD)
  --help                Show this help message

Examples:
  $0                           # Run all quality checks
  $0 --verbose                # Run with detailed output
  $0 --coverage-threshold 90  # Set higher coverage requirement

Quality Checks Performed:
  - Code coverage analysis
  - Static code analysis (ruff, mypy)
  - Security scanning (bandit, safety)
  - Code complexity analysis
  - Dependency analysis
  - Performance benchmarks
  - Documentation coverage
EOF
}

setup_environment() {
    log_info "Setting up quality check environment..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Install quality tools if not available
    local tools_needed=()
    
    if ! python3 -m pip list | grep -q pytest-cov; then
        tools_needed+=("pytest-cov")
    fi
    
    if ! command -v bandit &> /dev/null; then
        tools_needed+=("bandit[toml]")
    fi
    
    if ! command -v safety &> /dev/null; then
        tools_needed+=("safety")
    fi
    
    if ! command -v ruff &> /dev/null; then
        tools_needed+=("ruff")
    fi
    
    if ! command -v mypy &> /dev/null; then
        tools_needed+=("mypy")
    fi
    
    if [[ ${#tools_needed[@]} -gt 0 ]]; then
        log_info "Installing required tools: ${tools_needed[*]}"
        python3 -m pip install "${tools_needed[@]}"
    fi
}

run_coverage_analysis() {
    log_info "Running coverage analysis..."
    
    local coverage_file="$OUTPUT_DIR/coverage_${TIMESTAMP}.json"
    local coverage_html="$OUTPUT_DIR/coverage_html_${TIMESTAMP}"
    
    # Run tests with coverage
    pytest tests/ \
        --cov=src \
        --cov-report=json:"$coverage_file" \
        --cov-report=html:"$coverage_html" \
        --cov-report=term-missing \
        --cov-fail-under="$COVERAGE_THRESHOLD" \
        --tb=short \
        -q > "$OUTPUT_DIR/test_results_${TIMESTAMP}.txt" 2>&1
    
    # Extract coverage percentage
    local coverage_percent
    if [[ -f "$coverage_file" ]]; then
        coverage_percent=$(python3 -c "
import json
with open('$coverage_file') as f:
    data = json.load(f)
    print(f\"{data['totals']['percent_covered']:.1f}\")
" 2>/dev/null || echo "0")
    else
        coverage_percent="0"
    fi
    
    log_info "Code coverage: ${coverage_percent}% (threshold: ${COVERAGE_THRESHOLD}%)"
    
    # Check if coverage meets threshold
    if (( $(echo "$coverage_percent >= $COVERAGE_THRESHOLD" | bc -l) )); then
        echo "✅ Coverage check passed"
        return 0
    else
        echo "❌ Coverage check failed"
        return 1
    fi
}

run_static_analysis() {
    log_info "Running static code analysis..."
    
    local ruff_output="$OUTPUT_DIR/ruff_${TIMESTAMP}.json"
    local mypy_output="$OUTPUT_DIR/mypy_${TIMESTAMP}.json"
    
    # Run ruff linting
    log_debug "Running ruff linting..."
    ruff check src tests \
        --output-format=json \
        --output-file="$ruff_output" || true
    
    # Run ruff formatting check
    if ! ruff format --check src tests > "$OUTPUT_DIR/format_check_${TIMESTAMP}.txt" 2>&1; then
        log_warn "Code formatting issues found"
    fi
    
    # Run mypy type checking
    log_debug "Running mypy type checking..."
    mypy src \
        --ignore-missing-imports \
        --json-report "$OUTPUT_DIR/mypy_report_${TIMESTAMP}" \
        > "$mypy_output" 2>&1 || true
    
    # Count issues
    local ruff_issues=0
    local mypy_issues=0
    
    if [[ -f "$ruff_output" ]]; then
        ruff_issues=$(python3 -c "
import json
try:
    with open('$ruff_output') as f:
        data = json.load(f)
        print(len(data) if isinstance(data, list) else 0)
except:
    print(0)
")
    fi
    
    if [[ -f "$mypy_output" ]]; then
        mypy_issues=$(grep -c "error:" "$mypy_output" 2>/dev/null || echo "0")
    fi
    
    log_info "Ruff issues: $ruff_issues"
    log_info "MyPy issues: $mypy_issues"
    
    if [[ "$ruff_issues" -eq 0 && "$mypy_issues" -eq 0 ]]; then
        echo "✅ Static analysis passed"
        return 0
    else
        echo "❌ Static analysis found issues"
        return 1
    fi
}

run_security_analysis() {
    log_info "Running security analysis..."
    
    local bandit_output="$OUTPUT_DIR/bandit_${TIMESTAMP}.json"
    local safety_output="$OUTPUT_DIR/safety_${TIMESTAMP}.json"
    
    # Run bandit security scan
    log_debug "Running bandit security scan..."
    bandit -r src \
        -f json \
        -o "$bandit_output" \
        --severity-level medium || true
    
    # Run safety vulnerability check
    log_debug "Running safety vulnerability check..."
    safety check \
        --json \
        --output "$safety_output" || true
    
    # Count security issues
    local bandit_issues=0
    local safety_issues=0
    
    if [[ -f "$bandit_output" ]]; then
        bandit_issues=$(python3 -c "
import json
try:
    with open('$bandit_output') as f:
        data = json.load(f)
        print(len(data.get('results', [])))
except:
    print(0)
")
    fi
    
    if [[ -f "$safety_output" ]]; then
        safety_issues=$(python3 -c "
import json
try:
    with open('$safety_output') as f:
        data = json.load(f)
        print(len(data.get('vulnerabilities', [])))
except:
    print(0)
")
    fi
    
    log_info "Security issues (Bandit): $bandit_issues"
    log_info "Vulnerabilities (Safety): $safety_issues"
    
    local total_security_issues=$((bandit_issues + safety_issues))
    
    if [[ "$total_security_issues" -le "$SECURITY_THRESHOLD" ]]; then
        echo "✅ Security analysis passed"
        return 0
    else
        echo "❌ Security analysis found issues"
        return 1
    fi
}

run_complexity_analysis() {
    log_info "Running complexity analysis..."
    
    local complexity_output="$OUTPUT_DIR/complexity_${TIMESTAMP}.txt"
    
    # Use radon for complexity analysis if available
    if command -v radon &> /dev/null; then
        radon cc src --json > "$OUTPUT_DIR/complexity_${TIMESTAMP}.json"
        radon mi src --json > "$OUTPUT_DIR/maintainability_${TIMESTAMP}.json"
        
        # Calculate average complexity
        local avg_complexity
        avg_complexity=$(python3 -c "
import json
try:
    with open('$OUTPUT_DIR/complexity_${TIMESTAMP}.json') as f:
        data = json.load(f)
        total_complexity = 0
        function_count = 0
        for file_data in data.values():
            for item in file_data:
                if item['type'] in ['function', 'method']:
                    total_complexity += item['complexity']
                    function_count += 1
        avg = total_complexity / function_count if function_count > 0 else 0
        print(f'{avg:.2f}')
except:
    print('0.00')
")
        
        log_info "Average cyclomatic complexity: $avg_complexity"
        
        if (( $(echo "$avg_complexity <= $COMPLEXITY_THRESHOLD" | bc -l) )); then
            echo "✅ Complexity analysis passed"
            return 0
        else
            echo "❌ Complexity analysis failed"
            return 1
        fi
    else
        log_warn "Radon not available, installing..."
        python3 -m pip install radon
        run_complexity_analysis
    fi
}

run_dependency_analysis() {
    log_info "Running dependency analysis..."
    
    local dep_output="$OUTPUT_DIR/dependencies_${TIMESTAMP}.json"
    
    # Check for outdated packages
    pip list --outdated --format=json > "$dep_output" 2>/dev/null || echo "[]" > "$dep_output"
    
    # Count outdated dependencies
    local outdated_count
    outdated_count=$(python3 -c "
import json
try:
    with open('$dep_output') as f:
        data = json.load(f)
        print(len(data))
except:
    print(0)
")
    
    log_info "Outdated dependencies: $outdated_count"
    
    # Check for known vulnerabilities
    if [[ -f "$OUTPUT_DIR/safety_${TIMESTAMP}.json" ]]; then
        local vuln_count
        vuln_count=$(python3 -c "
import json
try:
    with open('$OUTPUT_DIR/safety_${TIMESTAMP}.json') as f:
        data = json.load(f)
        print(len(data.get('vulnerabilities', [])))
except:
    print(0)
")
        
        log_info "Vulnerable dependencies: $vuln_count"
    fi
    
    echo "✅ Dependency analysis completed"
    return 0
}

run_performance_benchmarks() {
    log_info "Running performance benchmarks..."
    
    local benchmark_output="$OUTPUT_DIR/benchmarks_${TIMESTAMP}.json"
    
    # Run performance tests if available
    if [[ -d "tests/performance" ]]; then
        pytest tests/performance/ \
            -v \
            --tb=short \
            --json-report \
            --json-report-file="$benchmark_output" || true
        
        echo "✅ Performance benchmarks completed"
    else
        log_warn "No performance tests found"
        echo "⚠️  Performance benchmarks skipped"
    fi
    
    return 0
}

check_documentation_coverage() {
    log_info "Checking documentation coverage..."
    
    local doc_output="$OUTPUT_DIR/documentation_${TIMESTAMP}.txt"
    
    # Count Python files and docstrings
    local python_files
    local documented_functions
    local total_functions
    
    python_files=$(find src -name "*.py" | wc -l)
    
    # Use a Python script to check docstring coverage
    python3 -c "
import ast
import os
import glob

total_functions = 0
documented_functions = 0

for py_file in glob.glob('src/**/*.py', recursive=True):
    try:
        with open(py_file, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                total_functions += 1
                if ast.get_docstring(node):
                    documented_functions += 1
    except:
        continue

doc_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
print(f'Documentation coverage: {doc_coverage:.1f}% ({documented_functions}/{total_functions})')

with open('$doc_output', 'w') as f:
    f.write(f'Python files: {python_files}\n')
    f.write(f'Total functions/classes: {total_functions}\n')
    f.write(f'Documented: {documented_functions}\n')
    f.write(f'Coverage: {doc_coverage:.1f}%\n')
"
    
    echo "✅ Documentation coverage checked"
    return 0
}

generate_quality_report() {
    if [[ "$GENERATE_REPORT" != "true" ]]; then
        return 0
    fi
    
    log_info "Generating quality report..."
    
    local report_file="$OUTPUT_DIR/quality_report_${TIMESTAMP}.html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Quality Report - Sentiment Analyzer Pro</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f4f4f4; padding: 20px; border-radius: 5px; }
        .metric { margin: 10px 0; padding: 10px; border-left: 4px solid #007cba; }
        .pass { border-left-color: #28a745; }
        .fail { border-left-color: #dc3545; }
        .warn { border-left-color: #ffc107; }
        .timestamp { color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Quality Report - Sentiment Analyzer Pro</h1>
        <p class="timestamp">Generated: $(date)</p>
    </div>
    
    <h2>Quality Metrics Summary</h2>
EOF
    
    # Add metrics to report
    if [[ -f "$OUTPUT_DIR/coverage_${TIMESTAMP}.json" ]]; then
        local coverage_percent
        coverage_percent=$(python3 -c "
import json
with open('$OUTPUT_DIR/coverage_${TIMESTAMP}.json') as f:
    data = json.load(f)
    print(f\"{data['totals']['percent_covered']:.1f}\")
" 2>/dev/null || echo "0")
        
        local coverage_class="pass"
        if (( $(echo "$coverage_percent < $COVERAGE_THRESHOLD" | bc -l) )); then
            coverage_class="fail"
        fi
        
        cat >> "$report_file" << EOF
    <div class="metric $coverage_class">
        <h3>Code Coverage</h3>
        <p><strong>${coverage_percent}%</strong> (threshold: ${COVERAGE_THRESHOLD}%)</p>
    </div>
EOF
    fi
    
    # Add other metrics...
    cat >> "$report_file" << EOF
    
    <h2>Detailed Results</h2>
    <ul>
        <li><a href="coverage_html_${TIMESTAMP}/index.html">Coverage Report</a></li>
        <li><a href="ruff_${TIMESTAMP}.json">Linting Results</a></li>
        <li><a href="bandit_${TIMESTAMP}.json">Security Scan</a></li>
        <li><a href="complexity_${TIMESTAMP}.json">Complexity Analysis</a></li>
    </ul>
    
    <h2>Recommendations</h2>
    <ul>
        <li>Review any failed quality checks above</li>
        <li>Address security issues immediately</li>
        <li>Improve test coverage for critical components</li>
        <li>Refactor complex functions (complexity > 10)</li>
    </ul>
    
</body>
</html>
EOF
    
    log_info "Quality report generated: $report_file"
}

cleanup() {
    log_debug "Cleaning up temporary files..."
    # Add cleanup logic if needed
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --verbose)
                VERBOSE=true
                shift
                ;;
            --no-report)
                GENERATE_REPORT=false
                shift
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --coverage-threshold)
                COVERAGE_THRESHOLD="$2"
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
    
    log_info "Starting quality check process..."
    log_info "Output directory: $OUTPUT_DIR"
    
    # Setup error handling
    trap cleanup EXIT
    
    # Setup environment
    setup_environment
    
    # Run quality checks
    local checks_passed=0
    local checks_failed=0
    
    echo
    echo "🔍 Running Quality Checks"
    echo "========================="
    
    if run_coverage_analysis; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi
    
    if run_static_analysis; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi
    
    if run_security_analysis; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi
    
    if run_complexity_analysis; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi
    
    if run_dependency_analysis; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi
    
    if run_performance_benchmarks; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi
    
    if check_documentation_coverage; then
        ((checks_passed++))
    else
        ((checks_failed++))
    fi
    
    # Generate report
    generate_quality_report
    
    # Summary
    echo
    echo "📊 Quality Check Summary"
    echo "======================="
    echo "Checks passed: $checks_passed"
    echo "Checks failed: $checks_failed"
    echo "Total checks: $((checks_passed + checks_failed))"
    
    if [[ "$checks_failed" -eq 0 ]]; then
        echo "🎉 All quality checks passed!"
        exit 0
    else
        echo "❌ Some quality checks failed"
        exit 1
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi