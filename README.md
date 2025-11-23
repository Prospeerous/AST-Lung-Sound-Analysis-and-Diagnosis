# Hybrid Lung Sound Analysis and Diagnosis
A hybrid deep learning system for automated respiratory disease diagnosis from lung sound recordings. The project implements a unique architecture combining unsupervised and supervised learning to classify audio into 5 disease categories and 4 sound types.

## Project Overview
This system analyzes lung sound recordings to identify respiratory conditions and abnormal sound patterns:

**Disease Categories (5 classes):**
- Asthma
- Bronchitis
- COPD (Chronic Obstructive Pulmonary Disease)
- Normal
- Pneumonia

**Sound Types (4 classes):**
- Normal
- Crackle
- Wheeze
- Both (Crackle + Wheeze)

**Models:**
- Hybrid Medical Neural Network combining:
  - Unsupervised Acoustic Encoder (VAE-based, 256D latent space)
  - Supervised Frequency Branch (CNN with vertical pattern detection)
  - Supervised Temporal Branch (CNN with horizontal pattern detection)
  - Feature Fusion Layer (4352D → 256D)
  - Dual Classification Heads (Sound + Disease)

**Performance:**
- Training uses multi-task learning with weighted loss
- Model validated on ICBHI 2017 Respiratory Sound Database
- Provides confidence scores for all predictions
- Early stopping with patience for optimal generalization

## Requirements
- Python 3.9+ (required)
- pip, setuptools, wheel
- 2GB+ RAM for inference
- 4GB+ RAM recommended for training
- GPU optional (speeds up training significantly)

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Prospeerous/AST-Lung-Sound-Analysis-and-Diagnosis.git
cd AST-Lung-Sound-Analysis-and-Diagnosis
```

### 2. Create and activate virtual environment

**Create virtual environment (all OS):**
```bash
python -m venv venv
```

Note: Ensure you have Python 3.9 or later installed. Check your version:
```bash
python --version
```

**Activate virtual environment:**

macOS/Linux:
```bash
source venv/bin/activate
```

Windows (Command Prompt):
```cmd
venv\Scripts\activate.bat
```

Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

Note: You should see `(venv)` in your terminal prompt when activated.

### 3. Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This will install:
- **Flask 3.0.0** - Web framework
- **PyTorch 2.0+** - Deep learning
- **Librosa 0.10.1** - Audio processing
- **PyOTP 2.9+** - Two-factor authentication
- **SQLAlchemy** - Database ORM
- And other dependencies

### 4. Initialize database

The application automatically creates the database on first run:
```bash
python app.py
```

The database will be created at `instance/lung_sound.db`.

Press Ctrl+C to stop the server after initialization.

### 5. Create admin account (optional)

To create an admin account, navigate to:
```
http://127.0.0.1:5000/create-admin?secret=LUNG_ADMIN_2024_SECRET_KEY
```

Default admin credentials (change immediately):
- **Email**: admin@lunganalysis.com
- **Password**: admin123

### 6. Verify trained model

Ensure the trained model exists at:
```
outputs/hybrid_model.pth
```

If missing, you need to train the model using the Jupyter notebook `Hybrid ast notebook.ipynb`.

### 7. Run the application

```bash
python app.py
```

Server runs at http://127.0.0.1:5000

The application will:
1. Initialize database tables (on first run)
2. Create upload and visualization directories
3. Load the hybrid model
4. Run diagnostic health checks
5. Start Flask development server

### 8. Access the web interface

**Register a new clinician account:**
1. Navigate to http://127.0.0.1:5000/register
2. Fill in username, email, phone, and password
3. Login and start analyzing lung sounds

**Upload and analyze audio:**
1. From dashboard, go to "Analysis" page
2. Enter patient information
3. Upload audio file (WAV, MP3, or FLAC)
4. View results with confidence scores and visualizations

## API Endpoints

### Public Routes
- `GET /` - Landing page
- `GET /login` - Login page
- `POST /login` - Process login
- `GET /register` - Registration page
- `POST /register` - Process registration
- `GET /logout` - Logout user

### Authentication Routes
- `GET /setup-2fa` - Two-factor authentication setup
- `POST /enable-2fa` - Enable 2FA for user
- `POST /disable-2fa` - Disable 2FA for user
- `GET /verify-2fa` - 2FA verification page
- `POST /verify-2fa` - Verify 2FA code

### Clinician Routes (Login Required)
- `GET /dashboard` - Clinician dashboard
- `GET /analysis` - Audio upload page
- `POST /analyze` - Process audio analysis
- `GET /results/<analysis_id>` - View analysis results
- `GET /patients` - List all patients
- `GET /patient/<patient_id>` - Patient detail view
- `GET /reports` - Diagnostic statistics
- `GET /settings` - User settings
- `GET /clinical-guidelines` - Medical reference guide

### Admin Routes
- `GET /admin/dashboard` - Admin overview
- `GET /admin/clinicians` - Manage clinicians
- `GET /admin/patients` - Manage all patients
- `POST /admin/reassign-patient` - Reassign patient to another clinician

## Project Structure

```
AST-Lung-Sound-Analysis-and-Diagnosis/
├── app.py                          # Main Flask application (52KB)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── add_2fa_fields.py              # Database migration for 2FA
│
├── database/
│   ├── __init__.py
│   └── models.py                   # SQLAlchemy models (User, Patient, Analysis)
│
├── templates/                      # HTML templates (Jinja2)
│   ├── base.html                   # Base template with navigation
│   ├── login.html                  # Login page
│   ├── register.html               # Registration form
│   ├── dashboard.html              # Clinician dashboard
│   ├── analysis.html               # Audio upload interface
│   ├── results.html                # Analysis results display
│   ├── patients.html               # Patient list view
│   ├── patient_detail.html         # Individual patient history
│   ├── reports.html                # Statistics and trends
│   ├── settings.html               # User settings
│   ├── clinical_guidelines.html    # Medical reference
│   ├── setup_2fa.html              # 2FA setup with QR code
│   ├── verify_2fa.html             # 2FA verification
│   ├── admin_dashboard.html        # Admin overview
│   ├── admin_clinicians.html       # Clinician management
│   └── admin_patients.html         # Patient management
│
├── static/
│   ├── uploads/                    # Uploaded audio files (git-ignored)
│   └── visualizations/             # Generated plots (git-ignored)
│
├── outputs/                        # Model artifacts
│   ├── hybrid_model.pth            # Trained hybrid model (required)
│   ├── dataset_metadata.csv        # Dataset statistics
│   └── preprocessing_summary.csv   # Preprocessing logs
│
├── instance/
│   └── lung_sound.db              # SQLite database (auto-created, git-ignored)
│
├── docs/                          # Additional documentation
│   ├── ARCHITECTURE.md            # System architecture details
│   ├── API.md                     # API reference
│   └── SCREENSHOTS_GUIDE.md       # Screenshot guide
│
├── .github/
│   ├── ISSUE_TEMPLATE/            # Bug report and feature request templates
│   └── workflows/
│       └── python-app.yml         # CI/CD workflow
│
├── Hybrid ast notebook.ipynb      # Model training notebook
├── CONTRIBUTING.md                # Contribution guidelines
├── CODE_OF_CONDUCT.md             # Community standards
├── CHANGELOG.md                   # Version history
├── SECURITY.md                    # Security policy
└── LICENSE                        # MIT License
```

## Model Architecture Details

### Hybrid Medical Network Components

**1. Unsupervised Acoustic Encoder (VAE-based)**
- 4 convolutional blocks (64 → 128 → 256 → 512 channels)
- Learns general acoustic features without labels
- 256-dimensional latent representation
- Trained on combined dataset using reconstruction + KL divergence loss
- Pre-trained and frozen during supervised training

**2. Supervised Frequency Branch**
- Custom FrequencyBlock with (3×1) convolutions
- 3 blocks with max pooling (32 → 64 → 128 channels)
- Analyzes vertical frequency patterns
- Captures pitch, harmonics, and frequency-domain abnormalities

**3. Supervised Temporal Branch**
- Custom TemporalBlock with (1×7) convolutions
- 3 blocks with max pooling (32 → 64 → 128 channels)
- Analyzes horizontal temporal patterns
- Captures rhythm, duration, and time-domain events

**4. Feature Fusion Layer**
- Concatenates unsupervised (256D) + supervised (4096D) features
- Two-layer MLP with dropout
- Linear(4352 → 512) + ReLU + Dropout(0.3)
- Linear(512 → 256) + ReLU + Dropout(0.2)
- Produces unified 256D representation

**5. Dual Classification Heads**
- Sound Classifier: Linear(256 → 4)
- Disease Classifier: Linear(256 → 5)
- Multi-task learning with weighted loss

### Model Input Specifications
- **Format**: Mel spectrogram (224×224 pixels)
- **Sample Rate**: 16,000 Hz
- **Duration**: 4 seconds (fixed)
- **Mel Bins**: 128
- **FFT Size**: 2048
- **Hop Length**: 256
- **Frequency Range**: 50-8000 Hz
- **Normalization**: Z-score (mean=0, std=1)

## Training the Model

If you need to train the model from scratch:

1. Open `Hybrid ast notebook.ipynb` in Jupyter
2. Ensure dataset is available in the correct location
3. Run all cells sequentially:
   - Data loading and preprocessing
   - Unsupervised encoder training (VAE)
   - Supervised branch training
   - Hybrid model training with frozen encoder
   - Model evaluation and visualization
4. Trained model will be saved to `outputs/hybrid_model.pth`

Training takes approximately:
- **Unsupervised phase**: 15 epochs (~30-45 minutes on CPU)
- **Hybrid phase**: 25 epochs with early stopping (~1-2 hours on CPU)
- **GPU**: Significantly faster (10-20 minutes total)

## Hardware Requirements
- **Minimum**: 2GB RAM, any CPU
- **Recommended**: 4GB+ RAM, multi-core CPU
- **GPU**: Optional (CUDA-compatible GPU speeds up training by 5-10x)
- **Developed on**: Windows 10/11, 8GB RAM, CPU-only

## Security Features

### Authentication & Access Control
- **Password Hashing**: Werkzeug with salt
- **Session Management**: Flask-Login
- **Two-Factor Authentication**: TOTP-based (RFC 6238)
- **Role-Based Access**: Clinician and Admin roles
- **Route Protection**: `@login_required` decorator

### Data Security
- **Input Validation**: File format and size checks
- **Secure Filenames**: `secure_filename()` sanitization
- **CSRF Protection**: Built-in Flask security
- **Database**: SQLite with SQLAlchemy ORM

### Best Practices
- Change default admin credentials immediately
- Enable 2FA for sensitive accounts
- Keep dependencies updated
- Use HTTPS in production
- Regular database backups

## Troubleshooting

**Python version error?**
```bash
python --version  # Should show 3.9.x or higher
```
Install Python 3.9+ from python.org if needed.

**Virtual environment not active?**

Check for `(venv)` in terminal prompt. If missing:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**Model not found error?**

Train the model using `Hybrid ast notebook.ipynb` or ensure `outputs/hybrid_model.pth` exists.

**Port already in use?**

Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5001 to any available port
```

**Audio upload fails?**

Check that audio is in supported format (WAV, MP3, FLAC) and under 50MB.

**Database error on first run?**

Delete `instance/lung_sound.db` and restart the application to recreate the database.

**Out of memory during training?**

Reduce batch size in the notebook training cells (default is 16, try 8 or 4).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick Start:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Follow PEP 8 style guidelines
4. Write meaningful comments explaining "why", not "what"
5. Test thoroughly before submitting
6. Submit a Pull Request

For bug reports and feature requests, use our [issue templates](.github/ISSUE_TEMPLATE/).

## For Reviewers

**Quick Start (No Training Required):**

If you have the pre-trained `hybrid_model.pth` file:

1. Unzip/clone the repository
2. Create virtual environment and activate
3. Install dependencies: `pip install -r requirements.txt`
4. Run application: `python app.py`
5. Access at http://127.0.0.1:5000
6. Register account and start analyzing

**With Training:**

If you need to train the model:
1. Open `Hybrid ast notebook.ipynb` in Jupyter
2. Run all cells (takes 1-2 hours on CPU)
3. Trained model saves to `outputs/hybrid_model.pth`
4. Run `python app.py`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design details.

## Documentation

Comprehensive documentation available:
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** - Community standards
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[SECURITY.md](SECURITY.md)** - Security policy
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture
- **[docs/API.md](docs/API.md)** - API reference
- **[docs/SCREENSHOTS_GUIDE.md](docs/SCREENSHOTS_GUIDE.md)** - Screenshot guide

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Contact

**Author**: Abigael Wambui Mwangi
**Email**: abigael.mwangi@strathmore.edu
**GitHub**: [Prospeerous](https://github.com/Prospeerous)
**Repository**: [AST-Lung-Sound-Analysis-and-Diagnosis](https://github.com/Prospeerous/AST-Lung-Sound-Analysis-and-Diagnosis)

For questions, support, or collaboration inquiries, please reach out via email or open an issue on GitHub.

---

## Acknowledgments

This project was developed as an academic initiative to advance automated respiratory disease diagnosis using deep learning.

**Special Thanks:**
- Open-source communities: PyTorch, Flask, Librosa
- ICBHI 2017 Respiratory Sound Database contributors
- Medical professionals who provided clinical guidance
- Strathmore University for academic support

## Disclaimer

**IMPORTANT**: This system is intended for **research and educational purposes only**. It is **NOT** a substitute for professional medical diagnosis. All diagnostic results should be reviewed and verified by qualified healthcare professionals before making any clinical decisions. Always consult with licensed medical practitioners for health-related concerns.

---

*Advancing healthcare technology through artificial intelligence*
