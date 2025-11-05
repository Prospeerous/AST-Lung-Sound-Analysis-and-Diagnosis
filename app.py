from sqlalchemy import func
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime

from database.models import db, User, Patient, AudioFile, AnalysisResult

import numpy as np
import torchaudio
from models.ast_model import ASTModel
import torch

# Load models once at startup
model_a = ASTModel(num_classes=4)
model_a.load_state_dict(torch.load("outputs/model_a_best.pth", map_location="cpu"))
model_a.eval()

model_b = ASTModel(num_classes=5)
model_b.load_state_dict(torch.load("outputs/model_b_best.pth", map_location="cpu"))
model_b.eval()

CLASS_LABELS_A = ['Normal', 'Crackle', 'Wheeze', 'Both']
DISEASE_LABELS = ['Asthma', 'COPD', 'Pneumonia', 'Bronchitis', 'Normal']

def predict_sound_class(filepath):
    audio, sr = torchaudio.load(filepath)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    target_sr = 22050
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
        sr = target_sr
    melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=128, n_fft=1024, hop_length=512)
    spec = melspec(audio)
    if spec.size(2) < 862:
        spec = torch.nn.functional.pad(spec, (0, 862 - spec.size(2)))
    else:
        spec = spec[:, :, :862]
    mean = spec.mean()
    std = spec.std()
    norm_spec = (spec - mean) / std
    input_tensor = norm_spec.unsqueeze(0)
    with torch.no_grad():
        logits = model_a(input_tensor)
        prob = torch.nn.functional.softmax(logits, dim=1)
        pred_idx = prob.argmax(dim=1).item()
        conf = 100 * prob.max().item()
    return CLASS_LABELS_A[pred_idx], conf

def predict_disease_diagnosis(filepath):
    audio, sr = torchaudio.load(filepath)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    target_sr = 22050
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
        sr = target_sr
    melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=128, n_fft=1024, hop_length=512)
    spec = melspec(audio)
    if spec.size(2) < 862:
        spec = torch.nn.functional.pad(spec, (0, 862 - spec.size(2)))
    else:
        spec = spec[:, :, :862]
    mean = spec.mean()
    std = spec.std()
    norm_spec = (spec - mean) / std
    input_tensor = norm_spec.unsqueeze(0)
    with torch.no_grad():
        logits = model_b(input_tensor)
        prob = torch.nn.functional.softmax(logits, dim=1)
        pred_idx = prob.argmax(dim=1).item()
        conf = 100 * prob.max().item()
    return DISEASE_LABELS[pred_idx], conf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-for-academic-project'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lung_sound.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def is_admin():
    return current_user.is_authenticated and current_user.role == 'admin'

def check_patient_access(patient):
    if is_admin():
        return True
    if patient.clinician_id != current_user.id:
        abort(403)
    return True

def get_clinician_patients():
    if is_admin():
        return Patient.query.all()
    return Patient.query.filter_by(clinician_id=current_user.id).all()

def get_clinician_analyses():
    if is_admin():
        return AnalysisResult.query.order_by(AnalysisResult.analysis_date.desc()).all()
    return AnalysisResult.query.join(AudioFile).join(Patient).filter(
        Patient.clinician_id == current_user.id
    ).order_by(AnalysisResult.analysis_date.desc()).all()

def require_admin():
    if not is_admin():
        abort(403)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
            return redirect(url_for('register'))
        new_user = User(username=username, email=email, password_hash=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/patients')
@login_required
def patients_list():
    patients = get_clinician_patients()
    return render_template('patients.html', patients=patients)

@app.route('/patient/<int:patient_id>')
@login_required
def patient_detail(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    check_patient_access(patient)
    analyses = AnalysisResult.query.join(AudioFile).filter(AudioFile.patient_id == patient_id).order_by(AnalysisResult.analysis_date.desc()).all()
    return render_template('patient_detail.html', patient=patient, analyses=analyses)

@app.route('/dashboard')
@login_required
def dashboard():
    recent_analyses = get_clinician_analyses()[:10]
    return render_template('dashboard.html', recent_analyses=recent_analyses)

@app.route('/upload', methods=['POST'])
@login_required
def upload_audio():
    try:
        patient_name = request.form.get('patient_name')
        patient_id = request.form.get('patient_id')
        age = request.form.get('age')
        gender = request.form.get('gender')
        recording_location = request.form.get('recording_location', 'chest')
        if not all([patient_name, patient_id]):
            return jsonify({'error': 'Patient name and ID are required'}), 400
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['audio_file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        patient = Patient.query.filter_by(patient_id=patient_id, clinician_id=current_user.id).first()
        if not patient:
            patient = Patient(
                patient_name=patient_name,
                patient_id=patient_id,
                age=int(age) if age else None,
                gender=gender,
                clinician_id=current_user.id
            )
            db.session.add(patient)
            db.session.commit()
        filename = secure_filename(file.filename)
        unique_filename = f"{current_user.id}_{patient_id}_{datetime.now().timestamp()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        audio_file = AudioFile(filename=filename, file_path=filepath, recording_location=recording_location, patient_id=patient.id)
        db.session.add(audio_file)
        db.session.commit()
        label, confidence = predict_sound_class(filepath)
        disease_label, disease_confidence = predict_disease_diagnosis(filepath)
        analysis = AnalysisResult(
            classification=label,
            confidence_score=confidence,
            disease_diagnosis=disease_label,
            disease_confidence=disease_confidence,
            audio_file_id=audio_file.id
        )
        db.session.add(analysis)
        db.session.commit()
        return jsonify({'success': True, 'analysis_id': analysis.id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<int:analysis_id>')
@login_required
def view_results(analysis_id):
    analysis = AnalysisResult.query.get_or_404(analysis_id)
    patient = analysis.audio_file.patient
    check_patient_access(patient)
    
    # DEBUG: Print to console to verify data exists
    print(f"Disease Diagnosis: {analysis.disease_diagnosis}")
    print(f"Disease Confidence: {analysis.disease_confidence}")
    print(f"Patient Name: {patient.patient_name}")
    
    results_data = {
        'disease_diagnosis': analysis.disease_diagnosis,
        'disease_confidence': analysis.disease_confidence,
        'clinical_interpretation': 'Chronic Inflammatory Airway Disease',  # Hardcoded for now
        'clinical_detail': 'A chronic condition characterized by inflammation and narrowing of the airways.',
        'patient_name': patient.patient_name,
        'patient_id': patient.patient_id,
        'age': patient.age,
        'gender': patient.gender,
        'filename': analysis.audio_file.filename,
        'recording_location': analysis.audio_file.recording_location,
        'analysis_date': analysis.analysis_date.strftime('%B %d, %Y at %I:%M %p'),
        'file_size': '3.2 MB',
        'classification': analysis.classification,
        'confidence_score': analysis.confidence_score,
        'spectrogram_path': None,
        'waveform_path': None,
        'attention_map_path': None
    }
    
    # DEBUG: Print what we're sending to template
    print(f"Results data keys: {results_data.keys()}")
    
    return render_template('results.html', **results_data)



@app.route('/reports')
@login_required
def reports():
    from sqlalchemy import func
    
    # Get total number of analyses for current user (clinician)
    total_analyses = AnalysisResult.query.join(AudioFile).join(Patient).filter(
        Patient.clinician_id == current_user.id
    ).count()
    
    # Count by disease diagnosis
    disease_counts = db.session.query(
        AnalysisResult.disease_diagnosis,
        func.count(AnalysisResult.disease_diagnosis)
    ).join(AudioFile).join(Patient).filter(
        Patient.clinician_id == current_user.id
    ).group_by(AnalysisResult.disease_diagnosis).all()
    
    # Convert to dictionary
    disease_stats = {disease: count for disease, count in disease_counts}
    
    # Prepare stats dictionary - CHANGED: Bronchiolitis → Bronchitis
    stats = {
        'total_analyses': total_analyses,
        'normal_count': disease_stats.get('Normal', 0),
        'asthma_count': disease_stats.get('Asthma', 0),
        'copd_count': disease_stats.get('COPD', 0),
        'pneumonia_count': disease_stats.get('Pneumonia', 0),
        'bronchiectasis_count': disease_stats.get('Bronchiectasis', 0),
        'urti_count': disease_stats.get('URTI', 0),
        'bronchitis_count': disease_stats.get('Bronchitis', 0)  # FIXED: Changed from bronchiolitis_count
    }
    
    # Get all analyses ordered by date
    analyses = AnalysisResult.query.join(AudioFile).join(Patient).filter(
        Patient.clinician_id == current_user.id
    ).order_by(AnalysisResult.analysis_date.desc()).all()
    
    return render_template('reports.html', stats=stats, analyses=analyses)



@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        new_username = request.form.get('username')
        if new_username:
            current_user.username = new_username
            db.session.commit()
            flash('Settings updated successfully!', 'success')
            return redirect(url_for('settings'))
    return render_template('settings.html')


@app.route('/admin')
@login_required
def admin_dashboard():
    require_admin()
    total_users = User.query.filter_by(role='clinician').count()
    total_patients = Patient.query.count()
    total_analyses = AnalysisResult.query.count()
    recent_analyses = AnalysisResult.query.order_by(AnalysisResult.analysis_date.desc()).limit(10).all()
    return render_template('admin_dashboard.html', total_users=total_users, total_patients=total_patients, total_analyses=total_analyses, recent_analyses=recent_analyses)


@app.route('/admin/patients')
@login_required
def admin_patients():
    require_admin()
    patients = Patient.query.all()
    clinicians = User.query.filter_by(role='clinician').all()
    return render_template('admin_patients.html', patients=patients, clinicians=clinicians)


@app.route('/admin/reassign-patient', methods=['POST'])
@login_required
def admin_reassign_patient():
    require_admin()
    patient_id = request.form.get('patient_id')
    new_clinician_id = request.form.get('clinician_id')
    patient = Patient.query.get_or_404(patient_id)
    old_clinician_name = patient.clinician.username
    patient.clinician_id = new_clinician_id
    db.session.commit()
    flash(f'Patient reassigned from Dr. {old_clinician_name} to new clinician', 'success')
    return redirect(url_for('admin_patients'))


@app.route('/create-admin')
def create_admin():
    admin = User.query.filter_by(email='admin@lunganalysis.com').first()
    if not admin:
        admin = User(username='System Admin', email='admin@lunganalysis.com', password_hash=generate_password_hash('admin123'), role='admin')
        db.session.add(admin)
        db.session.commit()
        return "Admin created! Email: admin@lunganalysis.com, Password: admin123"
    return "Admin already exists!"


@app.errorhandler(403)
def forbidden(e):
    flash('Access denied.', 'error')
    return redirect(url_for('dashboard'))


@app.errorhandler(404)
def not_found(e):
    flash('Not found.', 'error')
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    with app.app_context():
        print("Dropping old tables...")
        db.drop_all()
        print("Creating new tables...")
        db.create_all()
        print("✓ Database initialized with fresh schema")
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        print("✓ Upload folder ready")
    print("\n" + "=" * 50)
    print("Lung Sound Analysis System Starting...")
    print("=" * 50)
    print("Open your browser to: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True)
