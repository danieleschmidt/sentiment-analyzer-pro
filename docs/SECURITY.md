# Security Guide

This document outlines the security measures implemented in the Sentiment Analyzer Pro project and provides guidance for secure usage.

## Security Features

### Web API Security

#### Input Validation & Sanitization
- **Text Input Validation**: All text inputs are validated with length limits (1-10,000 characters)
- **XSS Protection**: Script tags, JavaScript URLs, and event handlers are automatically stripped
- **Content-Type Validation**: API endpoints require proper JSON content types
- **Pydantic Validation**: Strong type checking and validation using Pydantic v2

#### Rate Limiting
- **Per-IP Rate Limiting**: Maximum 100 requests per minute per IP address
- **In-Memory Tracking**: Sliding window rate limiting with automatic cleanup
- **429 Responses**: Proper HTTP status codes for rate limit violations

#### Security Headers
- **X-Content-Type-Options**: `nosniff` to prevent MIME type sniffing
- **X-Frame-Options**: `DENY` to prevent clickjacking
- **X-XSS-Protection**: Browser XSS protection enabled
- **Strict-Transport-Security**: HSTS header for secure connections
- **Content-Security-Policy**: Restrictive CSP with `default-src 'self'`

### CLI Security

#### File Operation Protection
- **Path Traversal Prevention**: Blocks `..` sequences in file paths
- **Absolute Path Restrictions**: Only allows temp directories for absolute paths
- **File Size Limits**: Maximum 100MB file size for CSV uploads
- **Data Volume Limits**: Maximum 1M rows per dataset
- **File Existence Validation**: Verifies file existence before processing

#### Error Handling
- **Secure Error Messages**: No sensitive information leaked in error responses
- **Proper Exit Codes**: Clear error reporting without exposing internals
- **Logging**: Security events are logged for monitoring

## Security Best Practices

### Deployment Security

1. **Environment Variables**: Always use environment variables for sensitive configuration
   ```bash
   export MODEL_PATH=/secure/path/to/model.joblib
   ```

2. **HTTPS Only**: Always deploy the web server behind HTTPS
   ```bash
   # Use a reverse proxy like nginx with SSL termination
   sentiment-cli serve --host 127.0.0.1 --port 5000
   ```

3. **Network Security**: Bind to localhost by default, use explicit host configuration for external access
   ```bash
   # Secure (localhost only)
   sentiment-cli serve
   
   # External access (use with caution)
   sentiment-cli serve --host 0.0.0.0
   ```

### Data Security

1. **Input Sanitization**: The system automatically sanitizes text inputs, but validate data sources
2. **File Permissions**: Ensure proper file permissions on model files and data
3. **Temporary Files**: Clean up temporary files after processing

### Monitoring & Alerting

1. **Rate Limit Monitoring**: Monitor for rate limit violations
2. **Security Log Analysis**: Review security events in application logs
3. **File Access Patterns**: Monitor for suspicious file access attempts

## Security Testing

The project includes comprehensive security tests in `tests/test_security.py`:

- Input validation and sanitization tests
- Rate limiting functionality tests
- Path traversal protection tests
- Security header verification tests
- Error handling security tests

Run security tests with:
```bash
pytest tests/test_security.py -v
```

## Reporting Security Issues

If you discover a security vulnerability, please:

1. **Do not** open a public issue
2. Email the maintainers directly with details
3. Allow time for assessment and patching
4. Follow responsible disclosure practices

## Security Audit Checklist

- [ ] Input validation on all user-provided data
- [ ] Rate limiting configured appropriately
- [ ] Security headers present on all responses
- [ ] File operations properly restricted
- [ ] Error messages don't leak sensitive information
- [ ] HTTPS enabled in production
- [ ] Environment variables used for configuration
- [ ] Security tests passing
- [ ] Pre-commit hooks enabled for secret scanning

## Dependencies

The project uses several security-focused tools:

- **detect-secrets**: Pre-commit hook for secret scanning
- **Pydantic**: Strong input validation and type checking
- **Flask**: Web framework with security best practices
- **pytest**: Comprehensive security test suite

Regular dependency updates and security scanning are performed as part of the CI/CD pipeline.