# Quality Gates Execution Report

**Generated**: 2025-08-09T05:37:41.613857  
**Overall Status**: ❌ FAILED  
**Pass Rate**: 5.9% (1/17)  
**Total Execution Time**: 0.05s  

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Gates | 17 |
| Passed | 1 |
| Failed | 16 |
| Pass Rate | 5.9% |
| Avg Gate Time | 0.00s |
| Max Gate Time | 0.02s |

## Detailed Results

### ❌ Python Environment Check

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python --version && pip --version`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Python Syntax Check

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -m py_compile src/*.py`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Core Model Tests

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -m pytest tests/test_models.py -v --tb=short`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Preprocessing Tests

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -m pytest tests/test_preprocessing.py -v --tb=short`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Internationalization Tests

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -m pytest tests/test_advanced_features.py::TestInternationalization -v --tb=short`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Import Structure Check

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -c "from src.models import build_nb_model; print('✓ Core imports work')"`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Configuration Validation

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -c "from src.config import Config; print(f'✓ Model path: {Config.MODEL_PATH}')"`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Web App Import Check

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -c "from src.webapp import app; print('✓ Flask app can be imported')"`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ CLI Import Check

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -c "from src.cli import main; print('✓ CLI can be imported')"`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Advanced Features Import

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -c "from src.i18n import t; from src.compliance import get_compliance_manager; print('✓ Advanced features importable')"`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Performance Modules Check

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -c "from src.advanced_caching import get_cache_manager; from src.auto_scaling_advanced import get_advanced_auto_scaler; print('✓ Performance modules work')"`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Research Framework Check

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -c "from src.quantum_research_framework import get_research_manager, setup_quantum_sentiment_experiment; print('✓ Research framework operational')"`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Security Modules Check

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -c "from src.security_hardening import get_threat_detector; from src.health_monitoring import get_health_monitor; print('✓ Security and monitoring ready')"`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ Data Pipeline Test

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -c "import pandas as pd; from src.preprocessing import clean_text; print('✓ Data pipeline functional')"`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ❌ ML Pipeline Test

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -c "from src.models import build_nb_model; model = build_nb_model(); print('✓ ML pipeline functional')"`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

### ✅ File System Check

- **Status**: PASSED
- **Execution Time**: 0.02s
- **Command**: `ls -la data/ && ls -la src/ && ls -la tests/ && echo '✓ File system accessible'`

### ❌ Resource Check

- **Status**: FAILED
- **Execution Time**: 0.00s
- **Command**: `source venv/bin/activate && python -c "import psutil; print(f'✓ Memory: {psutil.virtual_memory().percent}% used, CPU: {psutil.cpu_percent()}%')"`

**Error Details**:
```
/bin/sh: 1: source: not found
...
```

