# Changelog

## [0.1.0] - 2024-05-01
### Added
- Initial release with preprocessing, training, prediction and evaluation tools.
- Flask web server and CLI interface.

## [Unreleased]
### Added
- Cross-validation utilities.
- Dataset summary and preprocessing commands.
- Simple metrics endpoint with request logging in the web server.
- Bandit and Ruff checks with pip caching in CI.
- **Comprehensive Makefile for developer workflow**:
  - `make setup` for one-command development environment setup
  - `make test` for test suite with coverage reporting
  - `make lint`, `make format`, `make security` for code quality
  - `make dev` and `make serve` for development and production servers
  - `make build`, `make docker-build` for packaging and deployment
  - `make clean` for artifact cleanup
  - Comprehensive help system with `make help`
  - Color-coded output and graceful degradation
- **Enhanced CONTRIBUTING.md documentation**:
  - Quick start guide using Makefile
  - Comprehensive development workflow documentation
  - Pull request guidelines with quality gates
  - Advanced development patterns and best practices
- **Makefile test suite** ensuring reliable development workflow
- Comprehensive security enhancements:
  - Input validation and sanitization for web API
  - Rate limiting with per-IP tracking
  - Security headers (HSTS, CSP, X-Frame-Options, etc.)
  - Path traversal protection for file operations
  - File size limits and data volume validation
  - Comprehensive security test suite
- Structured logging system:
  - JSON formatted logs with configurable output
  - Security event logging with detailed context
  - Performance metrics tracking
  - API request logging with duration tracking
  - Comprehensive logging test suite

### Changed
- Docker container binds to localhost by default.
- Enhanced input validation using Pydantic v2 field validators
- Improved error handling with secure error messages

### Fixed
- CLI refactored into modular commands.
- Cross-validation bug when splitting folds.

### Security
- Added protection against XSS, script injection, and path traversal attacks
- Implemented rate limiting to prevent abuse
- Enhanced file operation security with size and path validation
