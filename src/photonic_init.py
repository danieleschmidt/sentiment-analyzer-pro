"""
Photonic-MLIR Bridge package initialization.
Isolated initialization to avoid dependency conflicts.
"""

# Photonic-MLIR Bridge components
try:
    from .photonic_mlir_bridge import (
        PhotonicComponent, PhotonicConnection, PhotonicCircuit,
        PhotonicComponentType, WavelengthBand, PhotonicCircuitBuilder,
        SynthesisBridge, create_simple_mzi_circuit
    )
    from .photonic_cli import PhotonicCLI
    __photonic_available__ = True
    __photonic_import_error__ = None
except ImportError as e:
    __photonic_available__ = False
    __photonic_import_error__ = str(e)
    # Create stub classes to avoid import errors
    PhotonicComponent = None
    PhotonicConnection = None
    PhotonicCircuit = None
    PhotonicComponentType = None
    WavelengthBand = None
    PhotonicCircuitBuilder = None
    SynthesisBridge = None
    create_simple_mzi_circuit = None
    PhotonicCLI = None

# Security components
try:
    from .photonic_security import (
        SecurityValidator, InputSanitizer, RateLimiter,
        validate_input, sanitize_input, check_rate_limit
    )
    __security_available__ = True
except ImportError as e:
    __security_available__ = False

# Monitoring components
try:
    from .photonic_monitoring import (
        get_monitor, record_synthesis_metrics,
        record_validation_metrics, record_security_event
    )
    __monitoring_available__ = True
except ImportError as e:
    __monitoring_available__ = False

# Optimization components
try:
    from .photonic_optimization import (
        get_optimizer, cached_synthesis, parallel_synthesis
    )
    __optimization_available__ = True
except ImportError as e:
    __optimization_available__ = False

__photonic_version__ = "1.0.0"

def get_photonic_status():
    """Get status of photonic module availability."""
    return {
        "photonic_available": __photonic_available__,
        "security_available": __security_available__,
        "monitoring_available": __monitoring_available__,
        "optimization_available": __optimization_available__,
        "import_error": __photonic_import_error__,
        "version": __photonic_version__
    }