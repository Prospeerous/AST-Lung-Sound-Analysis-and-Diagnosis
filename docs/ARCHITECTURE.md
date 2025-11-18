# System Architecture Documentation

This document provides an in-depth look at the architecture of the Hybrid Lung Sound Analysis and Diagnosis System.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Model Architecture](#model-architecture)
- [Data Flow](#data-flow)
- [Database Schema](#database-schema)
- [Security Architecture](#security-architecture)
- [Deployment Architecture](#deployment-architecture)

## Overview

The system follows a **three-tier architecture** pattern, separating concerns between presentation, business logic, and data persistence. It uses a **hybrid deep learning approach** combining unsupervised and supervised learning for robust medical diagnosis.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
│  │  Templates │  │    CSS     │  │    JavaScript (Minimal)│ │
│  │   (Jinja2) │  │ (Bootstrap)│  │                        │ │
│  └────────────┘  └────────────┘  └────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Flask Application (app.py)          │   │
│  │                                                      │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │   │
│  │  │ Auth Module │  │ Audio Module │  │ AI Module  │ │   │
│  │  │ - Login     │  │ - Upload     │  │ - Hybrid   │ │   │
│  │  │ - Register  │  │ - Preprocess │  │   Model    │ │   │
│  │  │ - 2FA       │  │ - Extract    │  │ - Inference│ │   │
│  │  └─────────────┘  └──────────────┘  └────────────┘ │   │
│  │                                                      │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │   │
│  │  │ Patient Mgmt│  │ Analysis     │  │ Reporting  │ │   │
│  │  │ - CRUD      │  │ - Results    │  │ - Stats    │ │   │
│  │  │ - History   │  │ - Viz        │  │ - Export   │ │   │
│  │  └─────────────┘  └──────────────┘  └────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                            │
│  ┌────────────────────────────────────────────────────┐     │
│  │              SQLAlchemy ORM                        │     │
│  │  ┌──────┐  ┌─────────┐  ┌───────────┐  ┌───────┐ │     │
│  │  │ User │  │ Patient │  │ AudioFile │  │Analysis│ │     │
│  │  │      │  │         │  │           │  │ Result │ │     │
│  │  └──────┘  └─────────┘  └───────────┘  └───────┘ │     │
│  └────────────────────────────────────────────────────┘     │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────┐     │
│  │              SQLite Database                       │     │
│  │              (lung_sound.db)                       │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### Presentation Layer
- **Purpose**: User interface and interaction
- **Technologies**: HTML5, Jinja2, Bootstrap CSS, minimal JavaScript
- **Responsibilities**:
  - Render dynamic content
  - Form validation (client-side)
  - Display results and visualizations
  - Responsive design for multiple devices

#### Application Layer
- **Purpose**: Business logic and orchestration
- **Technologies**: Flask, PyTorch, Librosa, PyOTP
- **Responsibilities**:
  - Route handling and request processing
  - Authentication and authorization
  - Audio preprocessing and feature extraction
  - Model inference and prediction
  - Data validation and sanitization
  - Session management
  - File upload handling

#### Data Layer
- **Purpose**: Data persistence and retrieval
- **Technologies**: SQLAlchemy ORM, SQLite
- **Responsibilities**:
  - Database connection management
  - CRUD operations
  - Query optimization
  - Data integrity constraints
  - Relationship management

## Model Architecture

### Hybrid Medical Neural Network

The model combines **unsupervised** and **supervised learning** in a unified architecture:

```
Input: Mel Spectrogram (1 × 224 × 224)
                │
        ┌───────┴───────┐
        ▼               ▼
   UNSUPERVISED    SUPERVISED
    BRANCH          BRANCHES
        │               │
        ▼               ▼
┌───────────────┐  ┌─────────────────────────┐
│   Acoustic    │  │  Frequency   Temporal   │
│   Encoder     │  │    Branch      Branch   │
│   (VAE-style) │  │                         │
│               │  │  ┌─────────┐ ┌────────┐ │
│  4 Conv       │  │  │ (3×1)   │ │ (1×7)  │ │
│  Layers       │  │  │ Conv    │ │ Conv   │ │
│  + FC         │  │  │ Blocks  │ │ Blocks │ │
│               │  │  └─────────┘ └────────┘ │
│  Output:      │  │                         │
│  256D         │  │  Output: 2048D + 2048D │
└───────┬───────┘  └───────────┬─────────────┘
        │                      │
        └──────────┬───────────┘
                   ▼
           ┌──────────────┐
           │ Concatenation│
           │    (4352D)   │
           └──────┬───────┘
                  ▼
           ┌──────────────┐
           │ Feature Fusion│
           │     MLP      │
           │  (4352→512→  │
           │     256D)    │
           └──────┬───────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│Sound Type    │    │   Disease    │
│Classifier    │    │  Classifier  │
│  (256→4)     │    │   (256→5)    │
└──────────────┘    └──────────────┘
   │                      │
   ▼                      ▼
Normal, Crackle,    Normal, Asthma,
Wheeze, Both       Bronchitis, COPD,
                   Pneumonia
```

### Model Components

#### 1. Unsupervised Acoustic Encoder
```python
Architecture:
- Conv2d(1→64, 4×4, stride=2)  → BatchNorm → ReLU
- Conv2d(64→128, 4×4, stride=2) → BatchNorm → ReLU
- Conv2d(128→256, 4×4, stride=2) → BatchNorm → ReLU
- Conv2d(256→512, 4×4, stride=2) → BatchNorm → ReLU
- AdaptiveAvgPool2d(4×4)
- Flatten → Linear(8192→512) → ReLU
- Linear(512→256)

Purpose: Learn general acoustic representations
Output: 256-dimensional latent features
```

#### 2. Supervised Frequency Branch
```python
Architecture:
- FrequencyBlock(1→32, kernel=(3×1))  → MaxPool2d
- FrequencyBlock(32→64, kernel=(3×1))  → MaxPool2d
- FrequencyBlock(64→128, kernel=(3×1)) → MaxPool2d
- Flatten → 2048D

Purpose: Capture vertical frequency patterns
Focus: Pitch, harmonics, frequency-domain abnormalities
```

#### 3. Supervised Temporal Branch
```python
Architecture:
- TemporalBlock(1→32, kernel=(1×7))  → MaxPool2d
- TemporalBlock(32→64, kernel=(1×7))  → MaxPool2d
- TemporalBlock(64→128, kernel=(1×7)) → MaxPool2d
- Flatten → 2048D

Purpose: Capture horizontal temporal patterns
Focus: Rhythm, duration, time-domain events
```

#### 4. Feature Fusion Layer
```python
Architecture:
- Concatenate [256D + 2048D + 2048D] = 4352D
- Linear(4352→512) → ReLU → Dropout(0.3)
- Linear(512→256) → ReLU → Dropout(0.2)

Purpose: Combine multi-scale features
Output: Unified 256D representation
```

#### 5. Classification Heads
```python
Sound Classifier:  Linear(256→4)  + Softmax
Disease Classifier: Linear(256→5)  + Softmax

Purpose: Multi-task prediction
Outputs: Probability distributions
```

### Input Preprocessing Pipeline

```
Raw Audio File (WAV/MP3/FLAC)
        ↓
Load with Librosa (sr=16000 Hz)
        ↓
Resample if needed → 16000 Hz
        ↓
Pad or Trim → 4 seconds (64000 samples)
        ↓
Compute Mel Spectrogram
  - n_mels: 128
  - n_fft: 2048
  - hop_length: 256
  - fmin: 50 Hz, fmax: 8000 Hz
        ↓
Convert to dB scale (log amplitude)
        ↓
Resize → 224×224 pixels
        ↓
Normalize (μ=0, σ=1)
        ↓
Add channel dimension → (1, 224, 224)
        ↓
Convert to PyTorch Tensor
        ↓
Ready for Model Input
```

## Data Flow

### Complete Analysis Workflow

```
┌─────────────────┐
│ User uploads    │
│ audio file      │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Flask receives  │
│ POST request    │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Validate file:  │
│ - Format check  │
│ - Size limit    │
│ - Secure name   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Save to uploads/│
│ directory       │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Preprocess:     │
│ - Load audio    │
│ - Resample      │
│ - Normalize     │
│ - Extract mel   │
│   spectrogram   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Model inference:│
│ - Load model    │
│ - Forward pass  │
│ - Get logits    │
│ - Apply softmax │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Generate viz:   │
│ - Waveform plot │
│ - Spectrogram   │
│ - Save as PNG   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Create Patient  │
│ record (if new) │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Save AudioFile  │
│ to database     │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Save Analysis   │
│ Result record   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Render results  │
│ template        │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Display to user:│
│ - Diagnosis     │
│ - Confidence    │
│ - Visualizations│
│ - Patient info  │
└─────────────────┘
```

## Database Schema

### Entity-Relationship Diagram

```
┌───────────────────┐
│      User         │
├───────────────────┤
│ • id (PK)         │
│ • username        │
│ • email (unique)  │
│ • password_hash   │
│ • phone           │
│ • two_fa_enabled  │
│ • two_fa_secret   │
│ • is_admin        │
│ • created_at      │
└─────────┬─────────┘
          │ 1
          │
          │ *
┌─────────┴─────────┐
│     Patient       │
├───────────────────┤
│ • id (PK)         │
│ • clinician_id(FK)│────────────┐
│ • patient_id      │            │
│ • first_name      │            │
│ • middle_name     │            │
│ • last_name       │            │
│ • dob             │            │
│ • gender          │            │
│ • phone           │            │
│ • email           │            │
│ • created_at      │            │
│ • updated_at      │            │
└─────────┬─────────┘            │
          │ 1                    │
          │                      │
          │ *                    │
┌─────────┴─────────┐            │
│    AudioFile      │            │
├───────────────────┤            │
│ • id (PK)         │            │
│ • patient_id (FK) │────────────┤
│ • filename        │            │
│ • file_path       │            │
│ • recording_date  │            │
│ • visit_type      │            │
│ • location        │            │
│ • notes           │            │
│ • uploaded_at     │            │
└─────────┬─────────┘            │
          │ 1                    │
          │                      │
          │ 1                    │
┌─────────┴─────────┐            │
│  AnalysisResult   │            │
├───────────────────┤            │
│ • id (PK)         │            │
│ • audio_file_id(FK│────────────┘
│ • disease_pred    │
│ • disease_conf    │
│ • sound_type_pred │
│ • sound_type_conf │
│ • all_disease_probs│
│ • all_sound_probs │
│ • waveform_path   │
│ • spectrogram_path│
│ • analyzed_at     │
└───────────────────┘
```

### Table Descriptions

**User Table**
- Stores clinician and admin accounts
- Passwords hashed with Werkzeug
- 2FA support with TOTP secrets

**Patient Table**
- Patient demographic information
- Linked to clinician who created record
- Soft foreign key constraints

**AudioFile Table**
- Uploaded lung sound recordings
- Clinical context (visit type, location, notes)
- Linked to patient

**AnalysisResult Table**
- AI model predictions and confidence scores
- References to visualization files
- One-to-one with AudioFile

## Security Architecture

### Authentication Flow

```
Login Request
     ↓
Check credentials
     ↓
Password valid? ──No──→ Reject
     ↓ Yes
2FA enabled? ──No──→ Create session → Success
     ↓ Yes
Prompt for TOTP code
     ↓
User enters code
     ↓
Verify with PyOTP
     ↓
Valid? ──No──→ Reject
     ↓ Yes
Create session → Success
```

### Access Control Matrix

| Resource | Public | Clinician | Admin |
|----------|--------|-----------|-------|
| Home Page | ✓ | ✓ | ✓ |
| Login/Register | ✓ | ✓ | ✓ |
| Dashboard | ✗ | ✓ (own) | ✓ (all) |
| Upload Audio | ✗ | ✓ | ✗ |
| View Patient | ✗ | ✓ (own) | ✓ (all) |
| Edit Patient | ✗ | ✓ (own) | ✓ (all) |
| Reassign Patient | ✗ | ✗ | ✓ |
| User Management | ✗ | ✗ | ✓ |
| System Stats | ✗ | ✗ | ✓ |

### Security Layers

1. **Network Layer**: HTTPS (in production)
2. **Application Layer**: CSRF protection, secure sessions
3. **Authentication**: Password hashing, 2FA
4. **Authorization**: Role-based access control
5. **Data Layer**: Parameterized queries (SQLAlchemy)
6. **File System**: Secure filename validation

## Deployment Architecture

### Development Setup
```
Local Machine
├── Python Virtual Environment
├── SQLite Database (local file)
├── Flask Development Server (port 5000)
└── Static Files (served by Flask)
```

### Production Recommendations
```
                    Internet
                       ↓
              ┌────────────────┐
              │  Load Balancer │
              │   (Optional)   │
              └────────┬───────┘
                       ↓
              ┌────────────────┐
              │  Reverse Proxy │
              │  (Nginx/Apache)│
              │  - SSL/TLS     │
              │  - Static files│
              └────────┬───────┘
                       ↓
              ┌────────────────┐
              │  WSGI Server   │
              │  (Gunicorn)    │
              └────────┬───────┘
                       ↓
              ┌────────────────┐
              │ Flask App      │
              │ (Multiple      │
              │  workers)      │
              └────────┬───────┘
                       ↓
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌────────────────┐          ┌────────────────┐
│ PostgreSQL/    │          │ File Storage   │
│ MySQL Database │          │ (S3/local)     │
└────────────────┘          └────────────────┘
```

### Scalability Considerations

**Vertical Scaling**
- Increase CPU for faster model inference
- Add RAM for larger model/batch processing
- Use GPU for significant speedup (5-10×)

**Horizontal Scaling**
- Load balance across multiple app instances
- Shared database (PostgreSQL recommended)
- Centralized file storage (S3, Azure Blob)
- Redis for session management

**Optimization Strategies**
- Model quantization (reduce precision)
- Batch processing for multiple files
- Caching frequent predictions
- Asynchronous task queue (Celery)
- Database connection pooling

## Technology Decisions

### Why Flask?
- Lightweight and flexible
- Easy to learn and deploy
- Rich ecosystem
- Good for prototypes and MVPs

### Why SQLite?
- Zero configuration
- Serverless (embedded)
- Perfect for development
- Easy to migrate to PostgreSQL

### Why PyTorch?
- Research-friendly
- Dynamic computation graphs
- Strong community
- Excellent for prototyping

### Why Librosa?
- Industry standard for audio
- Comprehensive feature extraction
- Well-documented
- Active development

## Future Architecture Enhancements

### Short Term
- [ ] Add Redis for caching
- [ ] Implement task queue (Celery)
- [ ] Add API layer (REST/GraphQL)
- [ ] Containerize with Docker

### Medium Term
- [ ] Microservices architecture
- [ ] Separate inference service
- [ ] Message queue (RabbitMQ/Kafka)
- [ ] Monitoring (Prometheus/Grafana)

### Long Term
- [ ] Kubernetes orchestration
- [ ] Multi-region deployment
- [ ] Real-time WebSocket support
- [ ] Federated learning for privacy

---

For questions about the architecture, contact: abigael.mwangi@strathmore.edu
