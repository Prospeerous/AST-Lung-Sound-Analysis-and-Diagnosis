# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

The security of this project is taken seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do NOT Disclose Publicly

Please do **not** open a public GitHub issue for security vulnerabilities. This could put users at risk.

### 2. Report Privately

Send a detailed report to: **abigael.mwangi@strathmore.edu**

Include in your report:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact and severity assessment
- Any suggested fixes (if available)
- Your contact information for follow-up

### 3. Response Timeline

You can expect:
- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next planned release

### 4. Disclosure Policy

- We will acknowledge your report within 48 hours
- We will confirm the vulnerability and determine its severity
- We will work on a fix and test it thoroughly
- We will release a patch and notify users
- After the fix is released and users have had time to update (typically 2 weeks), we will:
  - Publicly acknowledge the vulnerability
  - Credit you for the discovery (unless you prefer to remain anonymous)
  - Publish details in our security advisories

## Security Best Practices for Users

### For Deployment

#### 1. Use HTTPS in Production
Never deploy this application over plain HTTP in production:
```python
# Use a reverse proxy (Nginx/Apache) with SSL/TLS
# Or configure Flask-Talisman:
from flask_talisman import Talisman
Talisman(app, force_https=True)
```

#### 2. Change Default Credentials
If you create an admin account, **immediately change** the default password:
- Default: `admin123`
- Change to a strong password (12+ characters, mixed case, numbers, symbols)

#### 3. Enable Two-Factor Authentication
All users, especially admins, should enable 2FA:
- Go to Settings
- Enable 2FA
- Scan QR code with authenticator app
- Store backup codes securely

#### 4. Set Strong Secret Keys
In production, set a strong Flask secret key:
```python
import os
import secrets

# Generate a secure secret key
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(32)
```

#### 5. Use Environment Variables
Never hardcode sensitive information:
```bash
export FLASK_SECRET_KEY='your-secret-key-here'
export DATABASE_URL='your-database-url'
export ADMIN_EMAIL='admin@example.com'
```

#### 6. Secure File Uploads
Configure secure upload settings:
```python
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = '/secure/path/uploads/'
```

#### 7. Database Security
- Use PostgreSQL or MySQL in production (not SQLite)
- Enable database encryption at rest
- Use strong database passwords
- Restrict database access to application server only

#### 8. Network Security
- Use a firewall to restrict access
- Only expose necessary ports (443 for HTTPS, 80 for HTTP redirect)
- Use a VPN for admin access
- Implement rate limiting to prevent brute force attacks

#### 9. Regular Updates
```bash
# Keep dependencies updated
pip list --outdated
pip install --upgrade -r requirements.txt

# Check for security vulnerabilities
pip install safety
safety check
```

#### 10. Backup Data
- Regular database backups (daily recommended)
- Store backups in secure, separate location
- Test backup restoration periodically
- Encrypt backup files

### For Development

#### 1. Never Commit Secrets
- Don't commit `.env` files
- Don't commit database files
- Don't commit `config.py` with secrets
- Use `.gitignore` properly

#### 2. Code Review
- Review all pull requests for security issues
- Look for SQL injection vulnerabilities
- Check for XSS vulnerabilities
- Verify input validation

#### 3. Static Analysis
Run security linters:
```bash
pip install bandit
bandit -r . -f json -o bandit-report.json

pip install safety
safety check --json
```

## Known Security Considerations

### 1. Medical Data Privacy
This application processes **sensitive medical data**. Ensure compliance with:
- HIPAA (United States)
- GDPR (European Union)
- Local data protection regulations

**Recommendations:**
- Encrypt data at rest
- Encrypt data in transit (HTTPS)
- Implement access logging
- Regular security audits
- Data anonymization for research

### 2. Audio File Upload
Audio files are uploaded by users and must be handled carefully:
- File type validation (only WAV, MP3, FLAC)
- File size limits (50MB default)
- Secure filename handling (`werkzeug.utils.secure_filename`)
- Virus scanning (optional but recommended)

### 3. SQL Injection
We use SQLAlchemy ORM which prevents SQL injection by default. However:
- Never use raw SQL queries without parameterization
- Always validate and sanitize user input
- Use ORM query methods

### 4. Cross-Site Scripting (XSS)
Flask's Jinja2 template engine auto-escapes variables by default:
- Don't use `|safe` filter unless absolutely necessary
- Validate all user input
- Sanitize displayed data

### 5. Cross-Site Request Forgery (CSRF)
Flask provides CSRF protection:
- Use Flask-WTF for forms
- Include CSRF tokens in all forms
- Validate tokens on POST requests

### 6. Password Security
Passwords are hashed using Werkzeug's `generate_password_hash`:
- Uses PBKDF2 with SHA-256
- Automatically salted
- Never store plain text passwords

### 7. Session Security
- Sessions are signed with secret key
- Httponly cookies prevent XSS access
- Secure flag should be enabled in production (HTTPS)
- Implement session timeout

## Security Checklist for Deployment

Before deploying to production:

- [ ] Change all default credentials
- [ ] Set strong, random `SECRET_KEY`
- [ ] Enable HTTPS (SSL/TLS)
- [ ] Configure firewall rules
- [ ] Disable Flask debug mode
- [ ] Set secure cookie flags
- [ ] Implement rate limiting
- [ ] Set up monitoring and logging
- [ ] Enable 2FA for all admin accounts
- [ ] Restrict database access
- [ ] Configure CORS properly (if using API)
- [ ] Set appropriate file upload limits
- [ ] Implement backup strategy
- [ ] Review and minimize exposed endpoints
- [ ] Remove or secure test/debug routes
- [ ] Set appropriate file permissions
- [ ] Configure security headers (CSP, X-Frame-Options, etc.)
- [ ] Set up intrusion detection
- [ ] Plan incident response procedures

## Security Features Implemented

### Authentication
- ✅ Password hashing (PBKDF2-SHA256)
- ✅ Two-Factor Authentication (TOTP)
- ✅ Session management (Flask-Login)
- ✅ Login attempt tracking (basic)

### Authorization
- ✅ Role-based access control (Clinician/Admin)
- ✅ Resource ownership validation
- ✅ Route-level protection (`@login_required`)

### Input Validation
- ✅ File type validation
- ✅ File size limits
- ✅ Secure filename handling
- ✅ Form data validation

### Output Encoding
- ✅ Auto-escaping in templates
- ✅ JSON sanitization

### Data Protection
- ✅ Password hashing
- ✅ 2FA secret encryption
- ⚠️ Database encryption at rest (not implemented - user responsibility)
- ⚠️ HTTPS (not implemented - deployment responsibility)

## Vulnerability Disclosure Timeline

### Responsible Disclosure Process

1. **Day 0**: Vulnerability reported
2. **Day 1-2**: Initial response and acknowledgment
3. **Day 3-7**: Vulnerability confirmed and severity assessed
4. **Day 7-30**: Patch development and testing (depending on severity)
5. **Day 30-35**: Patch released, users notified
6. **Day 45-60**: Public disclosure (after users have time to update)

## Security Hall of Fame

We recognize and thank security researchers who responsibly disclose vulnerabilities:

*(No vulnerabilities reported yet)*

## Contact

For security concerns:
- **Email**: abigael.mwangi@strathmore.edu
- **Subject Line**: [SECURITY] Brief description

For general issues (non-security):
- Open an issue on [GitHub](https://github.com/Prospeerous/AST-Lung-Sound-Analysis-and-Diagnosis/issues)

---

**Note**: This is an academic/research project. For production medical use, conduct a professional security audit and penetration testing before deployment.
