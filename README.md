# Hybrid Lung Sound Analysis and Diagnosis System

A sophisticated deep learning-powered web application for automated respiratory disease diagnosis from lung sound recordings. This system combines unsupervised and supervised learning in a hybrid architecture to provide accurate, reliable clinical assessments.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [User Roles](#user-roles)
- [Supported Diagnoses](#supported-diagnoses)
- [Project Structure](#project-structure)
- [Security Features](#security-features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This project implements an advanced medical diagnostic system that analyzes lung sound recordings to identify respiratory conditions. The system uses a **Hybrid Medical Neural Network** that combines:
- **Unsupervised Learning**: An acoustic encoder that learns general audio features
- **Supervised Learning**: Specialized frequency and temporal analysis branches
- **Feature Fusion**: Intelligent combination of both learning approaches for robust predictions

The web application provides a complete clinical workflow including patient management, audio file upload and analysis, visualization of results, and comprehensive reporting capabilities.

## Key Features

### Clinical Capabilities
- **Automated Disease Diagnosis**: Classifies lung sounds into 5 respiratory conditions (Asthma, Bronchitis, COPD, Normal, Pneumonia)
- **Sound Type Classification**: Identifies abnormal sounds (Crackles, Wheezes, Both, or Normal)
- **Visual Analysis**: Generates waveform and mel spectrogram visualizations for each recording
- **Confidence Scoring**: Provides probability distributions across all diagnostic categories
- **Clinical Guidelines**: Built-in reference guide for respiratory conditions

### Patient Management
- **Comprehensive Patient Records**: Track patient demographics, contact information, and medical history
- **Multi-Patient Support**: Each clinician can manage multiple patients
- **Analysis History**: Complete audit trail of all diagnostic analyses per patient
- **Visit Tracking**: Record visit types (Initial Assessment, Follow-up, Screening) and clinical notes

### Security & Authentication
- **Two-Factor Authentication (2FA)**: Optional TOTP-based 2FA for enhanced account security
- **Role-Based Access Control**: Separate permissions for Clinicians and Administrators
- **Secure Password Management**: Industry-standard password hashing
- **Session Management**: Secure user sessions with Flask-Login

### Administrative Features
- **User Management**: Admin dashboard for managing clinician accounts
- **Patient Reassignment**: Ability to transfer patients between clinicians
- **System Analytics**: Overview of total users, patients, and analyses
- **Usage Reports**: Comprehensive reporting on diagnostic trends

### Audio Processing
- **Multiple Format Support**: Accepts WAV, MP3, and FLAC audio files
- **Automatic Preprocessing**: Noise reduction, resampling to 16kHz, and normalization
- **Mel Spectrogram Generation**: Converts audio to time-frequency representations optimized for the model
- **Fixed Duration Processing**: Standardizes audio clips to 4 seconds for consistent analysis

## System Architecture

The system follows a three-tier architecture:

1. **Presentation Layer**: Flask-based web interface with responsive HTML templates
2. **Application Layer**: Python business logic handling authentication, patient management, and audio processing
3. **Data Layer**: SQLite database with SQLAlchemy ORM for data persistence

### Workflow
```
Audio Upload → Preprocessing → Hybrid Model Inference → Visualization → Database Storage → Results Display
```

## Technology Stack

### Backend
- **Flask 3.0.0**: Web framework
- **Flask-Login 0.6.3**: User session management
- **Flask-SQLAlchemy 3.1.1**: Database ORM
- **SQLite**: Relational database

### Deep Learning
- **PyTorch 2.0+**: Deep learning framework
- **TorchVision 0.15+**: Vision utilities
- **Timm 0.9+**: Transformer architectures
- **Custom Hybrid Architecture**: Combines VAE-style unsupervised encoder with supervised CNN branches

### Audio Processing
- **Librosa 0.10.1**: Audio feature extraction
- **SoundFile 0.12.1**: Audio file I/O
- **NumPy 1.22+**: Numerical computations
- **SciPy 1.11.4**: Signal processing

### Security
- **PyOTP 2.9+**: TOTP-based two-factor authentication
- **QRCode 7.4+**: QR code generation for 2FA setup
- **Werkzeug 3.0.1**: Password hashing and security utilities

### Visualization
- **Matplotlib 3.7+**: Audio waveform and spectrogram plotting

## Installation

### Prerequisites
- **Python 3.9 or higher**
- **Git** (for cloning the repository)
- At least **2GB RAM** (for model inference)
- **GPU recommended** (but not required; CPU inference is supported)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Prospeerous/AST-Lung-Sound-Analysis-and-Diagnosis.git
cd AST-Lung-Sound-Analysis-and-Diagnosis
```

### Step 2: Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download the Trained Model
Ensure the trained hybrid model is present at:
```
outputs/hybrid_model.pth
```

This file should contain the model state dictionary with the key `model_state_dict`.

### Step 5: Initialize Database and Folders
The application will automatically create necessary folders on first run:
- `static/uploads/`: For audio file storage
- `static/visualizations/`: For generated plots
- `instance/`: For the SQLite database

### Step 6: Create Admin Account (Optional)
To create an admin account, navigate to:
```
http://127.0.0.1:5000/create-admin?secret=LUNG_ADMIN_2024_SECRET_KEY
```

Default admin credentials (change immediately after first login):
- **Email**: admin@lunganalysis.com
- **Password**: admin123

## Usage

### Starting the Application
```bash
python app.py
```

The application will:
1. Initialize the database (creating tables on first run)
2. Load the hybrid model
3. Run diagnostic tests to verify model health
4. Start the Flask development server on `http://127.0.0.1:5000`

### User Registration
1. Navigate to `http://127.0.0.1:5000/register`
2. Fill in the registration form:
   - Username
   - Email (cannot contain "admin" for regular users)
   - Phone number (exactly 10 digits)
   - Password
3. Click "Register"

### Login
1. Navigate to `http://127.0.0.1:5000/login`
2. Enter your email and password
3. If 2FA is enabled, you'll be prompted for a verification code

### Uploading Audio for Analysis
1. From the dashboard, navigate to the analysis page
2. Fill in the patient information form:
   - **Required**: First Name, Last Name, Patient ID (4 digits)
   - **Optional**: Middle Name, Date of Birth, Gender, Phone, Email
   - **Clinical Context**: Recording datetime, visit type, recording location, clinical notes
3. Upload an audio file (WAV, MP3, or FLAC)
4. Click "Upload and Analyze"

### Viewing Results
After analysis, you'll see:
- **Disease Diagnosis**: Primary diagnosis with confidence percentage
- **Sound Type**: Classification of abnormal sounds detected
- **Probability Distribution**: Confidence scores for all disease categories
- **Visualizations**: Waveform and mel spectrogram plots
- **Patient Information**: Demographics and recording details
- **Clinician Information**: Details of the clinician who performed the analysis

### Managing Patients
- View all patients assigned to you in the "Patients" section
- Click on a patient to see their complete analysis history
- Track diagnoses over time for follow-up visits

### Reports and Analytics
- Navigate to the "Reports" section for diagnostic statistics
- View breakdown of diagnoses by disease type
- Access complete analysis history with filtering options

### Enabling Two-Factor Authentication
1. Go to "Settings"
2. Click "Enable 2FA"
3. Scan the QR code with an authenticator app (Google Authenticator, Authy, etc.)
4. Enter the 6-digit verification code to confirm

## Model Architecture

### Hybrid Medical Network Components

#### 1. Unsupervised Acoustic Encoder
- **Type**: Variational Autoencoder (VAE) style encoder
- **Purpose**: Learns general acoustic features without labels
- **Architecture**:
  - 4 convolutional layers (64 → 128 → 256 → 512 channels)
  - Batch normalization and ReLU activation
  - Adaptive pooling to 4×4 spatial dimensions
  - Fully connected layers for latent representation (256 dimensions)
- **Output**: Latent features capturing fundamental audio characteristics

#### 2. Supervised Frequency Branch
- **Type**: Frequency-domain CNN
- **Purpose**: Analyzes vertical frequency patterns
- **Architecture**:
  - Custom FrequencyBlock with (3×1) convolutions
  - 3 blocks with max pooling (32 → 64 → 128 channels)
  - Captures pitch, harmonics, and frequency-domain abnormalities

#### 3. Supervised Temporal Branch
- **Type**: Time-domain CNN
- **Purpose**: Analyzes horizontal temporal patterns
- **Architecture**:
  - Custom TemporalBlock with (1×7) convolutions
  - 3 blocks with max pooling (32 → 64 → 128 channels)
  - Captures rhythm, duration, and time-domain events

#### 4. Feature Fusion Layer
- **Type**: Multi-layer perceptron (MLP)
- **Input**: Concatenated unsupervised (256D) + supervised (4096D) features
- **Architecture**:
  - Linear(4352 → 512) + ReLU + Dropout(0.3)
  - Linear(512 → 256) + ReLU + Dropout(0.2)
- **Output**: Unified 256D feature representation

#### 5. Dual Classification Heads
- **Sound Classifier**: Linear(256 → 4) for sound type classification
- **Disease Classifier**: Linear(256 → 5) for disease diagnosis

### Model Input Specifications
- **Format**: Mel spectrogram (224×224 pixels)
- **Audio Preprocessing**:
  - Sample rate: 16,000 Hz
  - Duration: 4 seconds (fixed)
  - Mel bins: 128
  - FFT size: 2048
  - Hop length: 256
  - Frequency range: 50-8000 Hz
  - Normalization: Z-score (zero mean, unit variance)

## User Roles

### Clinician
- **Primary users** of the system
- Can register and manage their own account
- Upload audio files and request analyses
- View results for their own patients only
- Manage patient records they created
- Access clinical guidelines and reports

### Administrator
- **System managers** with elevated privileges
- Cannot create admin accounts through registration (must use special endpoint)
- View all patients and analyses across all clinicians
- Reassign patients between clinicians
- Access system-wide analytics and usage statistics
- Manage clinician accounts
- Cannot perform patient diagnoses (clinician-only function)

## Supported Diagnoses

### Disease Categories (5 classes)
1. **Asthma**: Chronic inflammatory airway disease
2. **Bronchitis**: Inflammation of bronchial tubes
3. **COPD (Chronic Obstructive Pulmonary Disease)**: Progressive lung disease
4. **Normal**: Healthy respiratory system
5. **Pneumonia**: Lung infection with inflammation

### Sound Types (4 classes)
1. **Normal**: No abnormal sounds detected
2. **Crackle**: Discontinuous, explosive sounds (often in pneumonia, fibrosis)
3. **Wheeze**: Continuous, musical sounds (often in asthma, COPD)
4. **Both**: Presence of both crackles and wheezes

## Project Structure

```
AST-Lung-Sound-Analysis-and-Diagnosis/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── add_2fa_fields.py              # Database migration script for 2FA
│
├── database/
│   ├── __init__.py
│   └── models.py                   # SQLAlchemy data models
│
├── templates/                      # HTML templates
│   ├── base.html                   # Base template with navigation
│   ├── login.html                  # Login page
│   ├── register.html               # Registration page
│   ├── dashboard.html              # Clinician dashboard
│   ├── analysis.html               # Audio upload and analysis form
│   ├── results.html                # Analysis results display
│   ├── patients.html               # Patient list
│   ├── patient_detail.html         # Individual patient view
│   ├── reports.html                # Statistics and reports
│   ├── settings.html               # User settings
│   ├── clinical_guidelines.html    # Medical reference guide
│   ├── setup_2fa.html              # 2FA setup page
│   ├── verify_2fa.html             # 2FA verification page
│   ├── admin_dashboard.html        # Admin overview
│   ├── admin_clinicians.html       # Clinician management
│   └── admin_patients.html         # Patient management (admin)
│
├── static/
│   ├── uploads/                    # Uploaded audio files
│   └── visualizations/             # Generated plots (waveforms, spectrograms)
│
├── outputs/
│   ├── hybrid_model.pth            # Trained hybrid model (required)
│   ├── dataset_metadata.csv        # Dataset information
│   └── preprocessing_summary.csv   # Preprocessing statistics
│
├── instance/
│   └── lung_sound.db              # SQLite database (auto-created)
│
└── venv/                          # Virtual environment (not in Git)
```

## Security Features

### Authentication
- **Password Hashing**: Uses Werkzeug's `generate_password_hash` with salt
- **Session Management**: Flask-Login handles secure session cookies
- **CSRF Protection**: Built-in Flask security for form submissions

### Two-Factor Authentication (2FA)
- **TOTP Algorithm**: Time-based One-Time Password (RFC 6238)
- **QR Code Setup**: Easy enrollment via QR code scanning
- **Backup Secret**: Manual entry option with base32 secret key
- **Per-User Toggle**: Users can enable/disable 2FA independently

### Access Control
- **Route Protection**: `@login_required` decorator on sensitive routes
- **Role Verification**: Admin-only routes check `is_admin()` function
- **Patient Access Control**: Clinicians can only access their own patients
- **Admin Restrictions**: Admins cannot perform clinical diagnoses

### Data Validation
- **Input Sanitization**: `secure_filename()` for file uploads
- **Format Validation**: Checks for allowed audio formats (WAV, MP3, FLAC)
- **Size Limits**: Maximum upload size of 50MB
- **Required Fields**: Server-side validation of required form data

## Contributing

Contributions to improve the system are welcome! Here's how you can help:

### Reporting Issues
1. Check if the issue already exists in GitHub Issues
2. Provide detailed description including:
   - Steps to reproduce
   - Expected vs. actual behavior
   - Environment details (OS, Python version)
   - Error messages or logs

### Suggesting Features
1. Open a GitHub Issue with the `enhancement` label
2. Describe the feature and its benefits
3. Provide use cases and examples

### Code Contributions
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes following the existing code style
4. Test thoroughly (ensure existing functionality isn't broken)
5. Commit with descriptive messages
6. Push to your fork
7. Open a Pull Request with detailed description

### Code Style Guidelines
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused and modular
- Update documentation for new features

## License

This project is licensed under the **MIT License**.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:

- The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

See the LICENSE file for full details.

## Contact

**Author**: Abigael Wambui Mwangi

**Email**: abigael.mwangi@strathmore.edu

**GitHub**: [Prospeerous](https://github.com/Prospeerous)

For questions, support, or collaboration inquiries, please reach out via email or open an issue on GitHub.

---

## Acknowledgments

This project was developed as an academic initiative to advance automated respiratory disease diagnosis using deep learning. Special thanks to:

- The open-source communities behind PyTorch, Flask, and Librosa
- Contributors to the ICBHI 2017 Respiratory Sound Database
- Medical professionals who provided clinical guidance
- Strathmore University for academic support

## Disclaimer

**IMPORTANT**: This system is intended for research and educational purposes only. It is **NOT** a substitute for professional medical diagnosis. All diagnostic results should be reviewed and verified by qualified healthcare professionals before making any clinical decisions. Always consult with licensed medical practitioners for health-related concerns.

---

Made with dedication to advancing healthcare technology
