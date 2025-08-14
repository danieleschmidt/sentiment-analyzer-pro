# Security configuration
SECURE_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block'
}

# No hardcoded secrets - use environment variables
API_KEY = os.environ.get('API_KEY', '')
SECRET_KEY = os.environ.get('SECRET_KEY', '')

# Secure defaults
DEBUG = False
TESTING = False
