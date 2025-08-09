# Quality Gates Execution Report

**Generated**: 2025-08-09T05:38:24.496386  
**Overall Status**: ❌ FAILED  
**Pass Rate**: 82.4% (14/17)  
**Total Execution Time**: 35.41s  

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Gates | 17 |
| Passed | 14 |
| Failed | 3 |
| Pass Rate | 82.4% |
| Avg Gate Time | 2.08s |
| Max Gate Time | 7.98s |

## Detailed Results

### ✅ Python Environment Check

- **Status**: PASSED
- **Execution Time**: 0.43s
- **Command**: `source venv/bin/activate && python --version && pip --version`

### ❌ Python Syntax Check

- **Status**: FAILED
- **Execution Time**: 0.31s
- **Command**: `source venv/bin/activate && python -m py_compile src/*.py`

**Error Details**:
```
  File "src/schemas.py", line 1
    ```python
    ^
SyntaxError: invalid syntax
...
```

### ✅ Core Model Tests

- **Status**: PASSED
- **Execution Time**: 3.01s
- **Command**: `source venv/bin/activate && python -m pytest tests/test_models.py -v --tb=short`

### ✅ Preprocessing Tests

- **Status**: PASSED
- **Execution Time**: 7.98s
- **Command**: `source venv/bin/activate && python -m pytest tests/test_preprocessing.py -v --tb=short`

### ✅ Internationalization Tests

- **Status**: PASSED
- **Execution Time**: 3.17s
- **Command**: `source venv/bin/activate && python -m pytest tests/test_advanced_features.py::TestInternationalization -v --tb=short`

### ✅ Import Structure Check

- **Status**: PASSED
- **Execution Time**: 2.04s
- **Command**: `source venv/bin/activate && python -c "from src.models import build_nb_model; print('✓ Core imports work')"`

### ✅ Configuration Validation

- **Status**: PASSED
- **Execution Time**: 1.95s
- **Command**: `source venv/bin/activate && python -c "from src.config import Config; print(f'✓ Model path: {Config.MODEL_PATH}')"`

### ❌ Web App Import Check

- **Status**: FAILED
- **Execution Time**: 1.90s
- **Command**: `source venv/bin/activate && python -c "from src.webapp import app; print('✓ Flask app can be imported')"`

**Error Details**:
```
2025-08-09 05:38:09,520 - src.photonic_mlir_bridge - INFO - _initialize_base_operations:335 - Initialized 4 base photonic operations
2025-08-09 05:38:09,524 - src.photonic_monitoring - INFO - register_check:258 - Registered health check: system
2025-08-09 05:38:09,524 - src.photonic_monitoring - INFO - register_check:258 - Registered health check: memory
2025-08-09 05:38:09,524 - src.photonic_monitoring - INFO - register_check:258 - Registered health check: synthesis
2025-08-09 05:38:09,524 - sr...
```

### ❌ CLI Import Check

- **Status**: FAILED
- **Execution Time**: 2.05s
- **Command**: `source venv/bin/activate && python -c "from src.cli import main; print('✓ CLI can be imported')"`

**Error Details**:
```
2025-08-09 05:38:11,477 - src.photonic_mlir_bridge - INFO - _initialize_base_operations:335 - Initialized 4 base photonic operations
2025-08-09 05:38:11,480 - src.photonic_monitoring - INFO - register_check:258 - Registered health check: system
2025-08-09 05:38:11,481 - src.photonic_monitoring - INFO - register_check:258 - Registered health check: memory
2025-08-09 05:38:11,481 - src.photonic_monitoring - INFO - register_check:258 - Registered health check: synthesis
2025-08-09 05:38:11,481 - sr...
```

### ✅ Advanced Features Import

- **Status**: PASSED
- **Execution Time**: 2.08s
- **Command**: `source venv/bin/activate && python -c "from src.i18n import t; from src.compliance import get_compliance_manager; print('✓ Advanced features importable')"`

### ✅ Performance Modules Check

- **Status**: PASSED
- **Execution Time**: 2.04s
- **Command**: `source venv/bin/activate && python -c "from src.advanced_caching import get_cache_manager; from src.auto_scaling_advanced import get_advanced_auto_scaler; print('✓ Performance modules work')"`

### ✅ Research Framework Check

- **Status**: PASSED
- **Execution Time**: 2.05s
- **Command**: `source venv/bin/activate && python -c "from src.quantum_research_framework import get_research_manager, setup_quantum_sentiment_experiment; print('✓ Research framework operational')"`

### ✅ Security Modules Check

- **Status**: PASSED
- **Execution Time**: 2.26s
- **Command**: `source venv/bin/activate && python -c "from src.security_hardening import get_threat_detector; from src.health_monitoring import get_health_monitor; print('✓ Security and monitoring ready')"`

### ✅ Data Pipeline Test

- **Status**: PASSED
- **Execution Time**: 2.10s
- **Command**: `source venv/bin/activate && python -c "import pandas as pd; from src.preprocessing import clean_text; print('✓ Data pipeline functional')"`

### ✅ ML Pipeline Test

- **Status**: PASSED
- **Execution Time**: 1.97s
- **Command**: `source venv/bin/activate && python -c "from src.models import build_nb_model; model = build_nb_model(); print('✓ ML pipeline functional')"`

### ✅ File System Check

- **Status**: PASSED
- **Execution Time**: 0.01s
- **Command**: `ls -la data/ && ls -la src/ && ls -la tests/ && echo '✓ File system accessible'`

### ✅ Resource Check

- **Status**: PASSED
- **Execution Time**: 0.06s
- **Command**: `source venv/bin/activate && python -c "import psutil; print(f'✓ Memory: {psutil.virtual_memory().percent}% used, CPU: {psutil.cpu_percent()}%')"`

