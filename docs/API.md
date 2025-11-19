# API Documentation

This document describes the routes and endpoints available in the Hybrid Lung Sound Analysis System.

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [Public Routes](#public-routes)
- [Clinician Routes](#clinician-routes)
- [Admin Routes](#admin-routes)
- [Data Models](#data-models)
- [Error Handling](#error-handling)

## Overview

The application is built with Flask and follows a traditional server-side rendering approach. All routes return HTML pages (except for some utility endpoints that return JSON).

### Base URL
- Development: `http://127.0.0.1:5000`
- Production: Your deployed domain

### Authentication Method
- Session-based authentication using Flask-Login
- Optional Two-Factor Authentication (TOTP)

## Authentication

### Register New User

**Endpoint:** `POST /register`

**Description:** Create a new clinician account

**Request (Form Data):**
```
username: string (required, unique)
email: string (required, unique, cannot contain "admin")
phone: string (required, exactly 10 digits)
password: string (required)
```

**Response:**
- Success: Redirect to `/login` with success message
- Error: Redirect to `/register` with error message

**Validations:**
- Username must not already exist
- Email must be unique and not contain "admin"
- Phone must be exactly 10 digits
- Password must not be empty

---

### Login

**Endpoint:** `POST /login`

**Description:** Authenticate user

**Request (Form Data):**
```
email: string (required)
password: string (required)
```

**Response:**
- Success without 2FA: Redirect to `/dashboard`
- Success with 2FA: Redirect to `/verify-2fa`
- Error: Redirect to `/login` with error message

---

### Two-Factor Authentication Verification

**Endpoint:** `POST /verify-2fa`

**Description:** Verify TOTP code for 2FA

**Request (Form Data):**
```
verification_code: string (required, 6 digits)
```

**Response:**
- Success: Redirect to `/dashboard`
- Error: Redirect to `/verify-2fa` with error message

---

### Logout

**Endpoint:** `GET /logout`

**Description:** End user session

**Response:** Redirect to `/login`

## Public Routes

### Home Page

**Endpoint:** `GET /`

**Description:** Landing page

**Response:** Renders `index.html` or redirects to login

---

### Login Page

**Endpoint:** `GET /login`

**Description:** Display login form

**Response:** Renders `login.html`

---

### Registration Page

**Endpoint:** `GET /register`

**Description:** Display registration form

**Response:** Renders `register.html`

---

### Create Admin Account (Protected)

**Endpoint:** `GET /create-admin`

**Description:** Create admin account (requires secret parameter)

**Query Parameters:**
```
secret: string (must match "LUNG_ADMIN_2024_SECRET_KEY")
```

**Response:**
- Success: Creates admin account with default credentials
- Error: "Unauthorized" message

**Default Admin Credentials:**
```
Email: admin@lunganalysis.com
Password: admin123
Username: admin
```

⚠️ **Security Note:** Change password immediately after first login

## Clinician Routes

All clinician routes require `@login_required` decorator.

### Dashboard

**Endpoint:** `GET /dashboard`

**Description:** Main clinician dashboard

**Response:** Renders `dashboard.html` with:
- Total patients count
- Total analyses count
- Recent patients (last 5)
- Recent analyses (last 5)

---

### Analysis Page

**Endpoint:** `GET /analysis`

**Description:** Audio upload and analysis form

**Response:** Renders `analysis.html`

---

### Upload and Analyze Audio

**Endpoint:** `POST /upload`

**Description:** Upload audio file and get diagnosis

**Request (Multipart Form Data):**
```
audio: file (required, WAV/MP3/FLAC, max 50MB)
patient_id: string (required, 4 digits)
first_name: string (required)
middle_name: string (optional)
last_name: string (required)
dob: date (optional)
gender: string (optional)
phone: string (optional)
email: string (optional)
recording_datetime: datetime (optional)
visit_type: string (optional: "Initial Assessment", "Follow-up", "Screening")
recording_location: string (optional)
clinical_notes: text (optional)
```

**Response:**
- Success: Redirect to `/result/<audio_file_id>`
- Error: Redirect to `/analysis` with error message

**Processing Steps:**
1. Validate file format and size
2. Save file to uploads directory
3. Preprocess audio (resample, normalize)
4. Generate mel spectrogram
5. Run model inference
6. Generate visualizations
7. Create/update patient record
8. Save audio file record
9. Save analysis result

---

### View Analysis Result

**Endpoint:** `GET /result/<int:audio_file_id>`

**Description:** Display analysis results

**URL Parameters:**
- `audio_file_id`: Integer (required)

**Response:** Renders `results.html` with:
- Disease diagnosis and confidence
- Sound type classification and confidence
- Probability distributions
- Waveform and spectrogram visualizations
- Patient information
- Clinician information

**Authorization:** Clinician can only view results for their own patients

---

### List Patients

**Endpoint:** `GET /patients`

**Description:** View all patients for current clinician

**Response:** Renders `patients.html` with list of patients

---

### Patient Detail

**Endpoint:** `GET /patient/<int:patient_id>`

**Description:** View patient details and analysis history

**URL Parameters:**
- `patient_id`: Integer (required)

**Response:** Renders `patient_detail.html` with:
- Patient demographic information
- All audio files and analyses for patient
- Analysis history timeline

**Authorization:** Clinician can only view their own patients

---

### Edit Patient

**Endpoint:** `POST /patient/<int:patient_id>/edit`

**Description:** Update patient information

**URL Parameters:**
- `patient_id`: Integer (required)

**Request (Form Data):**
```
first_name: string (required)
middle_name: string (optional)
last_name: string (required)
dob: date (optional)
gender: string (optional)
phone: string (optional)
email: string (optional)
```

**Response:**
- Success: Redirect to `/patient/<patient_id>` with success message
- Error: Redirect to `/patient/<patient_id>` with error message

**Authorization:** Clinician can only edit their own patients

---

### Reports

**Endpoint:** `GET /reports`

**Description:** View diagnostic statistics and reports

**Response:** Renders `reports.html` with:
- Breakdown of diagnoses by disease type
- Breakdown by sound type
- Complete analysis history
- Summary statistics

**Authorization:** Clinician sees only their own data

---

### Clinical Guidelines

**Endpoint:** `GET /clinical-guidelines`

**Description:** View reference guide for respiratory conditions

**Response:** Renders `clinical_guidelines.html`

---

### Settings

**Endpoint:** `GET /settings`

**Description:** User settings page (2FA, profile)

**Response:** Renders `settings.html`

---

### Enable Two-Factor Authentication

**Endpoint:** `POST /enable-2fa`

**Description:** Enable 2FA for current user

**Response:**
- Success: Redirect to `/setup-2fa`
- Error: Error message

---

### Setup 2FA

**Endpoint:** `GET /setup-2fa`

**Description:** Display QR code for 2FA setup

**Response:** Renders `setup_2fa.html` with:
- QR code for authenticator app
- Base32 secret key (for manual entry)
- Verification form

---

### Confirm 2FA Setup

**Endpoint:** `POST /confirm-2fa`

**Description:** Verify and activate 2FA

**Request (Form Data):**
```
verification_code: string (required, 6 digits)
```

**Response:**
- Success: Redirect to `/settings` with success message
- Error: Redirect to `/setup-2fa` with error message

---

### Disable Two-Factor Authentication

**Endpoint:** `POST /disable-2fa`

**Description:** Disable 2FA for current user

**Response:**
- Success: Redirect to `/settings` with success message
- Error: Error message

## Admin Routes

All admin routes require `@login_required` and admin role check.

### Admin Dashboard

**Endpoint:** `GET /admin/dashboard`

**Description:** System-wide overview

**Response:** Renders `admin_dashboard.html` with:
- Total users count
- Total patients count
- Total analyses count
- Recent activity across all clinicians

**Authorization:** Admin only

---

### Manage Clinicians

**Endpoint:** `GET /admin/clinicians`

**Description:** View and manage all clinician accounts

**Response:** Renders `admin_clinicians.html` with:
- List of all clinicians
- Patient counts per clinician
- Analysis counts per clinician

**Authorization:** Admin only

---

### Manage Patients (Admin)

**Endpoint:** `GET /admin/patients`

**Description:** View and manage all patients

**Response:** Renders `admin_patients.html` with:
- List of all patients
- Assigned clinician for each
- Reassignment controls

**Authorization:** Admin only

---

### Reassign Patient

**Endpoint:** `POST /admin/reassign-patient`

**Description:** Transfer patient to different clinician

**Request (Form Data):**
```
patient_id: integer (required)
new_clinician_id: integer (required)
```

**Response:**
- Success: Redirect to `/admin/patients` with success message
- Error: Redirect to `/admin/patients` with error message

**Authorization:** Admin only

## Data Models

### User Model

```python
{
    "id": integer,
    "username": string,
    "email": string,
    "password_hash": string (hashed),
    "phone": string,
    "two_fa_enabled": boolean,
    "two_fa_secret": string (encrypted),
    "created_at": datetime
}
```

**Methods:**
- `is_admin()`: Returns true if email contains "admin"

---

### Patient Model

```python
{
    "id": integer,
    "clinician_id": integer (foreign key -> User.id),
    "patient_id": string (4 digits),
    "first_name": string,
    "middle_name": string (nullable),
    "last_name": string,
    "dob": date (nullable),
    "gender": string (nullable),
    "phone": string (nullable),
    "email": string (nullable),
    "created_at": datetime,
    "updated_at": datetime
}
```

**Relationships:**
- `clinician`: Many-to-One with User
- `audio_files`: One-to-Many with AudioFile

---

### AudioFile Model

```python
{
    "id": integer,
    "patient_id": integer (foreign key -> Patient.id),
    "filename": string,
    "file_path": string,
    "recording_datetime": datetime (nullable),
    "visit_type": string (nullable),
    "recording_location": string (nullable),
    "clinical_notes": text (nullable),
    "uploaded_at": datetime
}
```

**Relationships:**
- `patient`: Many-to-One with Patient
- `analysis_result`: One-to-One with AnalysisResult

---

### AnalysisResult Model

```python
{
    "id": integer,
    "audio_file_id": integer (foreign key -> AudioFile.id, unique),
    "disease_diagnosis": string,
    "disease_confidence": float,
    "sound_type_classification": string,
    "sound_type_confidence": float,
    "disease_probabilities": json (dict of all disease probabilities),
    "sound_probabilities": json (dict of all sound type probabilities),
    "waveform_path": string,
    "spectrogram_path": string,
    "analyzed_at": datetime
}
```

**Relationships:**
- `audio_file`: One-to-One with AudioFile

---

### Disease Classes

```python
[
    "Normal",
    "Asthma",
    "Bronchitis",
    "COPD",
    "Pneumonia"
]
```

---

### Sound Type Classes

```python
[
    "Normal",
    "Crackle",
    "Wheeze",
    "Both"
]
```

## Error Handling

### HTTP Status Codes

- `200 OK`: Request successful
- `302 Found`: Redirect (common for form submissions)
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

### Error Messages

Errors are displayed using Flask's `flash()` messages:

```python
# Error categories
flash("Error message", "error")    # Red
flash("Success message", "success") # Green
flash("Info message", "info")       # Blue
flash("Warning message", "warning") # Yellow
```

### Common Error Scenarios

**Authentication Errors:**
- Invalid credentials
- 2FA code incorrect
- Session expired

**Authorization Errors:**
- Accessing another clinician's patient
- Non-admin accessing admin routes

**Validation Errors:**
- Invalid file format
- File size exceeded
- Missing required fields
- Invalid patient ID format

**Processing Errors:**
- Audio preprocessing failed
- Model inference failed
- File save failed
- Database error

## Rate Limiting

Currently **not implemented**. Recommended for production:

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route("/upload", methods=["POST"])
@limiter.limit("10 per hour")
def upload():
    # ...
```

## Future API Enhancements

Planned features:

- [ ] RESTful JSON API endpoints
- [ ] API authentication tokens (JWT)
- [ ] Batch audio processing endpoint
- [ ] Webhook notifications
- [ ] Export endpoints (PDF, CSV)
- [ ] Real-time analysis status via WebSocket
- [ ] Pagination for list endpoints
- [ ] Advanced filtering and search
- [ ] API versioning (/api/v1/...)

## Example Usage

### cURL Examples

**Login:**
```bash
curl -X POST http://localhost:5000/login \
  -d "email=user@example.com&password=mypassword" \
  -c cookies.txt
```

**Upload Audio (with session):**
```bash
curl -X POST http://localhost:5000/upload \
  -b cookies.txt \
  -F "audio=@/path/to/audio.wav" \
  -F "patient_id=1234" \
  -F "first_name=John" \
  -F "last_name=Doe" \
  -F "visit_type=Initial Assessment"
```

### Python Example

```python
import requests

# Login
session = requests.Session()
login_data = {
    "email": "user@example.com",
    "password": "mypassword"
}
session.post("http://localhost:5000/login", data=login_data)

# Upload audio
files = {"audio": open("audio.wav", "rb")}
data = {
    "patient_id": "1234",
    "first_name": "John",
    "last_name": "Doe",
    "visit_type": "Initial Assessment"
}
response = session.post("http://localhost:5000/upload", files=files, data=data)
print(response.url)  # Redirected to result page
```

---

For questions about the API, contact: abigael.mwangi@strathmore.edu
