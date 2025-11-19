# Changelog

All notable changes to the Hybrid Lung Sound Analysis and Diagnosis System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-11-17

### Added
- **Hybrid Neural Network Architecture**
  - Unsupervised acoustic encoder using VAE-style architecture
  - Supervised frequency branch for vertical pattern analysis
  - Supervised temporal branch for horizontal pattern analysis
  - Feature fusion layer combining both learning approaches
  - Dual classification heads (disease + sound type)

- **Web Application Features**
  - Flask-based web interface with responsive design
  - User authentication and session management
  - Role-based access control (Clinician and Admin roles)
  - Patient management system
  - Audio file upload and analysis interface
  - Results visualization (waveforms and mel spectrograms)
  - Analysis history tracking per patient
  - Comprehensive reporting and analytics dashboard

- **Security Features**
  - Two-Factor Authentication (2FA) using TOTP
  - QR code generation for 2FA setup
  - Secure password hashing with Werkzeug
  - Session-based authentication with Flask-Login
  - CSRF protection for form submissions
  - Access control for patient data

- **Clinical Features**
  - 5-class disease diagnosis (Asthma, Bronchitis, COPD, Normal, Pneumonia)
  - 4-class sound type classification (Normal, Crackle, Wheeze, Both)
  - Confidence scoring for all predictions
  - Clinical guidelines reference section
  - Patient visit tracking (Initial, Follow-up, Screening)
  - Recording location and clinical notes capture

- **Audio Processing**
  - Multi-format support (WAV, MP3, FLAC)
  - Automatic resampling to 16kHz
  - Noise reduction and normalization
  - Mel spectrogram generation (224×224)
  - Fixed duration processing (4 seconds)
  - Visual waveform and spectrogram generation

- **Administrative Tools**
  - Admin dashboard with system statistics
  - User management interface
  - Patient reassignment between clinicians
  - System-wide analytics and reporting

- **Documentation**
  - Comprehensive README with installation guide
  - Detailed architecture documentation
  - API and model specifications
  - Security features documentation
  - Contributing guidelines
  - MIT License

### Technical Stack
- **Backend**: Flask 3.0.0, SQLAlchemy, SQLite
- **Deep Learning**: PyTorch 2.0+, Timm, Custom hybrid architecture
- **Audio**: Librosa 0.10.1, SoundFile, NumPy
- **Security**: PyOTP, QRCode, Werkzeug
- **Visualization**: Matplotlib

## [Unreleased]

### Planned Features
- [ ] Automated testing suite (unit and integration tests)
- [ ] REST API for programmatic access
- [ ] PDF report generation and export
- [ ] Multi-language support (i18n)
- [ ] Dark mode theme
- [ ] Mobile-responsive enhancements
- [ ] Real-time audio analysis capability
- [ ] Model performance metrics dashboard
- [ ] Batch audio processing
- [ ] Advanced filtering and search in patient records
- [ ] Email notifications for analysis completion
- [ ] Audit logging for compliance
- [ ] Docker containerization
- [ ] Cloud deployment guides (AWS, Azure, GCP)

### Known Issues
- Model file (hybrid_model.pth) must be manually obtained
- Database migrations require manual script execution
- Limited to 50MB file upload size
- CPU inference can be slow for large batches
- No automated backup system for database

## Version History

### Version Naming Convention
- **Major version (X.0.0)**: Breaking changes, major new features
- **Minor version (0.X.0)**: New features, non-breaking changes
- **Patch version (0.0.X)**: Bug fixes, minor improvements

---

## [0.9.0] - 2024-11-15 (Pre-release)

### Added
- Initial implementation of 2FA authentication system
- Clinical guidelines reference page
- Enhanced UI with improved navigation
- Recording metadata capture (location, visit type, notes)

### Changed
- Improved error handling for audio processing
- Enhanced patient detail view with analysis history
- Updated database schema for 2FA fields

### Fixed
- Fixed issue with audio file validation
- Resolved session timeout problems
- Corrected timezone handling in timestamps

## [0.8.0] - 2024-11-10 (Beta)

### Added
- Patient management system
- Analysis results storage in database
- Report generation functionality
- Admin dashboard

### Changed
- Migrated from simple file storage to SQLAlchemy ORM
- Improved audio preprocessing pipeline
- Enhanced visualization generation

## [0.5.0] - 2024-11-01 (Alpha)

### Added
- Basic web interface for audio upload
- User registration and authentication
- Initial hybrid model integration
- Simple results display

### Changed
- Switched from prototype to Flask production setup
- Improved model loading and inference

## [0.1.0] - 2024-10-27 (Initial Development)

### Added
- Project initialization
- Basic Flask application structure
- Model architecture definition
- Database models
- Initial templates

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## Questions?

For questions about releases or features, please:
- Check the [README.md](README.md)
- Open an issue on [GitHub](https://github.com/Prospeerous/AST-Lung-Sound-Analysis-and-Diagnosis/issues)
- Contact: abigael.mwangi@strathmore.edu

---

**Note**: This project is for research and educational purposes. Always consult qualified healthcare professionals for medical decisions.
