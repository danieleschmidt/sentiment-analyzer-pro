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

### Changed
- Docker container binds to localhost by default.

### Fixed
- CLI refactored into modular commands.
- Cross-validation bug when splitting folds.
