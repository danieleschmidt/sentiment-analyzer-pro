# 🧭 Project Vision

> A short 2–3 sentence description of what this repo does, for whom, and why.

# 📅 12-Week Roadmap

## I1
- **Themes**: Security, Performance
- **Goals / Epics**
    - Harden API endpoints and CLI input handling
    - Optimize preprocessing for larger datasets
- **Definition of Done**
    - Input schema validation in webapp & CLI
    - Benchmarks show 20% speed improvement for cleaning functions

## I2
- **Themes**: Observability, Developer UX
- **Goals / Epics**
    - Add structured logging and metrics export
    - Simplify test execution and local setup
- **Definition of Done**
    - Metrics available at `/metrics` and exported to Prometheus
    - `make test` runs lint + pytest in <3m with no flakes

## I3
- **Themes**: Advanced Modeling, Release Prep
- **Goals / Epics**
    - Experiment with transformer fine-tuning
    - Prepare Docker and CI for production release
- **Definition of Done**
    - Transformer model evaluated and documented
    - CI builds and pushes versioned container images

# ✅ Epic & Task Checklist

### 🔐 Increment 1: Security & Refactoring
- [ ] [EPIC] Eliminate hardcoded secrets
  - [ ] Load from environment securely
  - [ ] Add `pre-commit` hook for scanning secrets
- [ ] [EPIC] Improve CI stability
  - [ ] Replace flaky integration tests
  - [ ] Enable parallel test execution

### 📈 Increment 2: Observability & Developer UX
- [ ] [EPIC] Structured logging
  - [ ] Standard JSON logs in CLI and webapp
  - [ ] Document log levels
- [ ] [EPIC] Local developer setup
  - [ ] `make setup` installs optional deps
  - [ ] Update CONTRIBUTING with quickstart

### 🤖 Increment 3: Advanced Modeling & Release
- [ ] [EPIC] Transformer fine-tuning
  - [ ] Add training script for BERT models
  - [ ] Compare results with baseline
- [ ] [EPIC] Production-ready container
  - [ ] Multi-stage Docker build
  - [ ] Push images from CI

# ⚠️ Risks & Mitigation
- Limited data volume → augment with public datasets
- Heavy ML dependencies slow CI → use lightweight test fixtures
- Incomplete NLTK downloads → cache datasets in CI image
- Library updates may break API → pin versions & run regression tests

# 📊 KPIs & Metrics
- [ ] >85% test coverage
- [ ] <15 min CI pipeline time
- [ ] <5% error rate on core service
- [ ] 100% secrets loaded from vault/env

# 👥 Ownership & Roles (Optional)
- DevOps: CI/CD pipeline, Docker images
- ML Engineer: modeling experiments
- QA: automated tests and coverage monitoring
