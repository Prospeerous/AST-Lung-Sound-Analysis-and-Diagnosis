import io
import os
import base64
import traceback
from datetime import datetime

import pyotp
import qrcode
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display

from sqlalchemy import func
from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from flask import flash
from flask import jsonify
from flask import abort
from flask import session
from flask_login import LoginManager
from flask_login import login_user
from flask_login import logout_user
from flask_login import login_required
from flask_login import current_user
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

from database.models import db
from database.models import User
from database.models import Patient
from database.models import AudioFile
from database.models import AnalysisResult

# ==================== Hybrid Model Architecture ====================
class FrequencyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 7), padding=(0, 3))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class AcousticEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc_mu = nn.Linear(512 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16, latent_dim)

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

class HybridMedicalNet(nn.Module):
    def __init__(self, num_sound_classes=4, num_disease_classes=5, latent_dim=256):
        super().__init__()
        self.unsupervised_encoder = AcousticEncoder(latent_dim)

        self.freq_branch = nn.Sequential(
            FrequencyBlock(1, 32),
            nn.MaxPool2d((2, 2)),
            FrequencyBlock(32, 64),
            nn.MaxPool2d((2, 2)),
            FrequencyBlock(64, 128),
            nn.MaxPool2d((2, 2)),
        )

        self.temporal_branch = nn.Sequential(
            TemporalBlock(1, 32),
            nn.MaxPool2d((2, 2)),
            TemporalBlock(32, 64),
            nn.MaxPool2d((2, 2)),
            TemporalBlock(64, 128),
            nn.MaxPool2d((2, 2)),
        )

        self.supervised_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(latent_dim + 256 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.sound_classifier = nn.Linear(256, num_sound_classes)
        self.disease_classifier = nn.Linear(256, num_disease_classes)

    def forward(self, x):
        # Unsupervised features
        unsup_mu, unsup_logvar = self.unsupervised_encoder(x)
        unsup_features = unsup_mu

        # Supervised features
        freq_features = self.freq_branch(x)
        temporal_features = self.temporal_branch(x)
        supervised_features = torch.cat([freq_features, temporal_features], dim=1)
        supervised_features = self.supervised_conv(supervised_features)
        supervised_features = supervised_features.view(supervised_features.size(0), -1)

        # Feature fusion
        combined_features = torch.cat([unsup_features, supervised_features], dim=1)
        fused_features = self.feature_fusion(combined_features)

        # Classifications
        sound_output = self.sound_classifier(fused_features)
        disease_output = self.disease_classifier(fused_features)

        return sound_output, disease_output

# ==================== Load Hybrid Model ====================
print("Loading Hybrid Medical Audio Model (hybrid_model.pth)...")

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_MODEL_PATH = os.path.join(BASE_DIR, 'outputs', 'hybrid_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DISEASE_LABELS = ['Asthma', 'Bronchitis', 'COPD', 'Normal', 'Pneumonia']
SOUND_LABELS = ['Both', 'Crackle', 'Normal', 'Wheeze']

print(f"[OK] Using device: {DEVICE}")
print(f"[OK] Hybrid model path: {HYBRID_MODEL_PATH}")

def load_hybrid_model():
    """Load the hybrid medical audio model"""
    print(f"Loading hybrid model from: {HYBRID_MODEL_PATH}")

    if not os.path.exists(HYBRID_MODEL_PATH):
        print(f"ERROR: HYBRID MODEL NOT FOUND: {HYBRID_MODEL_PATH}")
        return None

    print(f"SUCCESS: Hybrid model file found")

    try:
        model = HybridMedicalNet(num_sound_classes=4, num_disease_classes=5)
        print("SUCCESS: Hybrid model architecture created")

        checkpoint = torch.load(HYBRID_MODEL_PATH, map_location=DEVICE, weights_only=False)
        print(f"SUCCESS: Checkpoint loaded, keys: {list(checkpoint.keys())}")

        if 'model_state_dict' not in checkpoint:
            print("ERROR: 'model_state_dict' not found in checkpoint")
            return None

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        print(f"SUCCESS: Hybrid model loaded successfully on {DEVICE}")

        # Test hybrid model
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 224, 224).to(DEVICE)
            sound_out, disease_out = model(dummy_input)
            print(f"SUCCESS: Hybrid model test PASSED: sound={sound_out.shape}, disease={disease_out.shape}")

        return model

    except Exception as e:
        print(f"ERROR: Failed to load hybrid model: {e}")
        traceback.print_exc()
        return None

# Load hybrid model at startup
HYBRID_MODEL = load_hybrid_model()

if HYBRID_MODEL is None:
    print("CRITICAL: Hybrid model failed to load!")
    exit(1)
else:
    print("SUCCESS: HYBRID MODEL READY!")

# Mel spectrogram parameters for hybrid model
MEL_PARAMS = {
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 256,
    "target_sr": 16000,
    "clip_duration": 4,
}

print(f"[OK] Disease classes: {DISEASE_LABELS}")
print(f"[OK] Sound classes: {SOUND_LABELS}")
print(f"[OK] Mel parameters: {MEL_PARAMS}")

# ==================== Hybrid Model Health Diagnosis ====================
print("\n" + "=" * 70)
print("DIAGNOSING HYBRID MODEL HEALTH - CHECKING FOR COLLAPSE")
print("=" * 70)

print("\nTest: Predictions on 10 random inputs with hybrid model")
print("-" * 70)
random_predictions = []
with torch.no_grad():
    for i in range(10):
        random_input = torch.randn(1, 1, 224, 224).to(DEVICE)
        sound_logits, disease_logits = HYBRID_MODEL(random_input)
        disease_probs = torch.softmax(disease_logits, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(disease_probs)
        pred_conf = disease_probs[pred_class] * 100
        random_predictions.append(pred_class)
        print(f"  Random Test {i+1:2d}: {DISEASE_LABELS[pred_class]:12s} ({pred_conf:5.1f}%)")

unique_predictions = len(set(random_predictions))
most_common = max(set(random_predictions), key=random_predictions.count)
most_common_count = random_predictions.count(most_common)

print("\n" + "-" * 70)
print(f"Hybrid Model Diversity Analysis:")
print(f"   Unique classes predicted: {unique_predictions} out of {len(DISEASE_LABELS)}")
print(f"   Most common prediction: {DISEASE_LABELS[most_common]} ({most_common_count}/10 times)")

if unique_predictions == 1:
    print("\n[WARNING] CRITICAL WARNING: Hybrid model is COLLAPSED!")
    print("   [X] Model predicts ONLY ONE CLASS for all inputs")
    print("   [X] This model CANNOT be fixed without retraining")
    print("   [X] Predictions will be UNRELIABLE")
elif unique_predictions <= 2:
    print("\n[WARNING] WARNING: Hybrid model has LIMITED diversity")
    print("   [!] Model may be biased toward certain classes")
    print("   [!] Retraining recommended for better performance")
elif most_common_count >= 7:
    print("\n[WARNING] WARNING: Hybrid model shows STRONG BIAS")
    print(f"   [!] Predicts {DISEASE_LABELS[most_common]} in {most_common_count*10}% of cases")
    print("   [!] Retraining recommended with better class balancing")
else:
    print("\n[OK] Hybrid model appears healthy with good prediction diversity")
    print("   [OK] Unsupervised + Supervised features working well")

print("\n" + "=" * 70)
print("Hybrid model diagnosis complete. Starting Flask application...")
print("=" * 70 + "\n")

# ==================== Hybrid Model Prediction Functions ====================
def preprocess_audio_for_hybrid(filepath):
    """
    Preprocess audio file to mel spectrogram for Hybrid Model
    Returns normalized mel spectrogram as PyTorch tensor or None if processing fails
    """
    try:
        print(f"\n=== Starting audio preprocessing for hybrid model: {os.path.basename(filepath)} ===")

        if not os.path.exists(filepath):
            print(f"ERROR: File does not exist: {filepath}")
            return None

        print(f"Loading audio with sr={MEL_PARAMS['target_sr']}...")
        audio, sr = librosa.load(filepath, sr=MEL_PARAMS["target_sr"], mono=True)

        if audio is None or len(audio) == 0:
            print("ERROR: Audio file is empty or couldn't be loaded")
            return None

        print(f"[OK] Audio loaded: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f} seconds)")

        # Trim or pad to fixed duration
        target_length = MEL_PARAMS["clip_duration"] * MEL_PARAMS["target_sr"]
        if len(audio) > target_length:
            audio = audio[:target_length]
            print(f"[OK] Truncated audio to {MEL_PARAMS['clip_duration']} seconds")
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
            print(f"[OK] Padded audio to {MEL_PARAMS['clip_duration']} seconds")

        print(f"Generating mel spectrogram for hybrid model (n_mels={MEL_PARAMS['n_mels']}, n_fft={MEL_PARAMS['n_fft']})...")
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=MEL_PARAMS["n_mels"],
            n_fft=MEL_PARAMS["n_fft"],
            hop_length=MEL_PARAMS["hop_length"],
            fmin=50,
            fmax=8000,
        )

        if mel_spec.size == 0 or mel_spec.shape[0] == 0 or mel_spec.shape[1] == 0:
            print(f"ERROR: Mel spectrogram is empty. Shape: {mel_spec.shape}")
            return None

        print(f"[OK] Mel spectrogram generated: {mel_spec.shape}")

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Check for NaN or Inf values
        if np.isnan(mel_spec_db).any() or np.isinf(mel_spec_db).any():
            print("WARNING: Mel spectrogram contains NaN or Inf values, replacing with 0")
            mel_spec_db = np.nan_to_num(mel_spec_db, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"Mel spec dB range: [{mel_spec_db.min():.2f}, {mel_spec_db.max():.2f}]")

        # Normalize
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        print(f"Normalizing for hybrid model: mean={mean:.4f}, std={std:.4f}...")

        if std > 0:
            mel_spec_normalized = (mel_spec_db - mean) / std
        else:
            mel_spec_normalized = mel_spec_db - mean

        if np.isnan(mel_spec_normalized).any() or np.isinf(mel_spec_normalized).any():
            print("WARNING: Normalized spectrogram contains NaN or Inf, replacing with 0")
            mel_spec_normalized = np.nan_to_num(mel_spec_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"Normalized spec range: [{mel_spec_normalized.min():.2f}, {mel_spec_normalized.max():.2f}]")

        # Convert to tensor and resize to 224x224 for hybrid model
        spec_tensor = torch.from_numpy(mel_spec_normalized).float().unsqueeze(0).unsqueeze(0)
        spec_tensor = F.interpolate(spec_tensor, size=(224, 224), mode='bilinear', align_corners=False)

        print(f"[OK] Final feature shape for hybrid model: {spec_tensor.shape}")
        print("=== Audio preprocessing for hybrid model completed successfully ===\n")

        return spec_tensor

    except Exception as e:
        print(f"ERROR in audio preprocessing for hybrid model: {str(e)}")
        traceback.print_exc()
        return None

def predict_disease_diagnosis(filepath):
    """
    Predict respiratory disease diagnosis using Hybrid Model
    Returns: dict with prediction results or error
    """
    try:
        print(f"\n{'='*60}")
        print(f"Starting HYBRID disease diagnosis for: {os.path.basename(filepath)}")
        print(f"{'='*60}")

        # Preprocess audio for hybrid model
        spec_tensor = preprocess_audio_for_hybrid(filepath)

        if spec_tensor is None:
            error_msg = "Failed to extract features from audio for hybrid model. Audio file may be corrupted or in an unsupported format."
            print(f"[ERROR] {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "disease_label": "Unknown",
                "disease_confidence": 0.0,
            }

        if spec_tensor.numel() == 0:
            error_msg = "Mel spectrogram is empty after preprocessing for hybrid model"
            print(f"[ERROR] {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "disease_label": "Unknown",
                "disease_confidence": 0.0,
            }

        print("\nMaking prediction with HYBRID Model (Unsupervised + Supervised)...")

        HYBRID_MODEL.eval()
        with torch.no_grad():
            spec_tensor = spec_tensor.to(DEVICE)
            sound_logits, disease_logits = HYBRID_MODEL(spec_tensor)
            disease_probabilities = torch.softmax(disease_logits, dim=1)
            disease_probs = disease_probabilities.cpu().numpy()[0]

            sound_probabilities = torch.softmax(sound_logits, dim=1)
            sound_probs = sound_probabilities.cpu().numpy()[0]

        print("[OK] Hybrid prediction completed")
        print(f"Disease prediction shape: {disease_probs.shape}")
        print(f"Disease prediction values: {disease_probs}")

        if disease_probs is None or disease_probs.size == 0:
            error_msg = "Hybrid model returned empty prediction"
            print(f"[ERROR] {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "disease_label": "Unknown",
                "disease_confidence": 0.0,
            }

        print("\nProcessing hybrid disease prediction...")
        print(f"Disease probabilities from hybrid model: {disease_probs}")

        disease_idx = np.argmax(disease_probs)
        disease_confidence = float(disease_probs[disease_idx] * 100)
        disease_label = DISEASE_LABELS[disease_idx] if disease_idx < len(DISEASE_LABELS) else "Unknown"

        print(f"[OK] HYBRID Disease: {disease_label} (confidence: {disease_confidence:.2f}%)")

        sound_idx = np.argmax(sound_probs)
        sound_confidence = float(sound_probs[sound_idx] * 100)
        sound_label = SOUND_LABELS[sound_idx] if sound_idx < len(SOUND_LABELS) else "Unknown"

        print(f"[OK] HYBRID Sound type: {sound_label} (confidence: {sound_confidence:.2f}%)")

        all_disease_probabilities = {
            DISEASE_LABELS[i]: float(disease_probs[i] * 100)
            for i in range(min(len(DISEASE_LABELS), len(disease_probs)))
        }

        all_sound_probabilities = {
            SOUND_LABELS[i]: float(sound_probs[i] * 100)
            for i in range(min(len(SOUND_LABELS), len(sound_probs)))
        }

        print(f"\n{'='*60}")
        print("FINAL HYBRID DIAGNOSIS:")
        print(f"  Sound Type: {sound_label} ({sound_confidence:.2f}%)")
        print(f"  Disease: {disease_label} ({disease_confidence:.2f}%)")
        print(f"  Disease probabilities: {all_disease_probabilities}")
        print(f"  Sound probabilities: {all_sound_probabilities}")
        print(f"  Model: Hybrid (Unsupervised + Supervised)")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "error": None,
            "disease_label": disease_label,
            "disease_confidence": disease_confidence,
            "sound_label": sound_label,
            "sound_confidence": sound_confidence,
            "all_probabilities": all_disease_probabilities,
            "all_sound_probabilities": all_sound_probabilities,
        }

    except Exception as e:
        error_msg = f"Hybrid prediction error: {str(e)}"
        print(f"\nERROR: {error_msg}")
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg,
            "disease_label": "Unknown",
            "disease_confidence": 0.0,
        }

# ==================== Flask App Setup ====================
app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-for-academic-project"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///lung_sound.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["VISUALIZATIONS_FOLDER"] = "static/visualizations"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac"}

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def is_admin():
    return current_user.is_authenticated and current_user.role == "admin"

def generate_waveform_plot(audio_path, output_path, sr=MEL_PARAMS["target_sr"]):
    """Generates and saves a waveform plot."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#4A90E2")
        plt.title("Audio Waveform", fontsize=14, fontweight="bold")
        plt.xlabel("Time (s)", fontsize=11)
        plt.ylabel("Amplitude", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[OK] Waveform plot saved to {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Error generating waveform plot: {e}")
        traceback.print_exc()
        return False

def generate_spectrogram_plot(audio_path, output_path, sr=MEL_PARAMS["target_sr"], n_mels=MEL_PARAMS["n_mels"], n_fft=MEL_PARAMS["n_fft"], hop_length=MEL_PARAMS["hop_length"]):
    """Generates and saves a mel spectrogram plot."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
        plt.colorbar(format="%+2.0f dB", label="Intensity (dB)")
        plt.title("Mel Spectrogram (Frequency Analysis)", fontsize=14, fontweight="bold")
        plt.xlabel("Time (s)", fontsize=11)
        plt.ylabel("Frequency (Hz)", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[OK] Spectrogram plot saved to {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Error generating spectrogram plot: {e}")
        traceback.print_exc()
        return False

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
    return (
        AnalysisResult.query.join(AudioFile)
        .join(Patient)
        .filter(Patient.clinician_id == current_user.id)
        .order_by(AnalysisResult.analysis_date.desc())
        .all()
    )

def require_admin():
    if not is_admin():
        abort(403)

# ==================== Routes ====================
@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        phone_number = request.form.get("phone_number")
        password = request.form.get("password")

        if "admin" in email.lower():
            flash("Invalid email address. Admin accounts cannot be created through registration.", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "error")
            return redirect(url_for("register"))

        if not phone_number or not phone_number.isdigit() or len(phone_number) != 10:
            flash("Phone number must be exactly 10 digits.", "error")
            return redirect(url_for("register"))

        new_user = User(
            username=username,
            email=email,
            phone_number=phone_number,
            password_hash=generate_password_hash(password),
            role="clinician",
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            if user.is_2fa_enabled:
                session["pending_2fa_user_id"] = user.id
                return redirect(url_for("verify_2fa"))
            else:
                login_user(user)
                flash("Login successful!", "success")
                return redirect(url_for("dashboard"))
        flash("Invalid email or password", "error")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.pop("pending_2fa_user_id", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/setup-2fa", methods=["GET", "POST"])
@login_required
def setup_2fa():
    """Setup page for enabling 2FA"""
    if request.method == "POST":
        verification_code = request.form.get("verification_code")
        totp = pyotp.TOTP(current_user.totp_secret)
        if totp.verify(verification_code):
            current_user.is_2fa_enabled = True
            db.session.commit()
            flash("Two-Factor Authentication has been enabled successfully!", "success")
            return redirect(url_for("settings"))

        else:
            flash("Invalid verification code. Please try again.", "error")
            totp_secret = current_user.totp_secret
    else:
        if not current_user.totp_secret:
            totp_secret = pyotp.random_base32()
            current_user.totp_secret = totp_secret
            db.session.commit()
        else:
            totp_secret = current_user.totp_secret

    totp_uri = pyotp.totp.TOTP(totp_secret).provisioning_uri(
        name=current_user.email, issuer_name="Hybrid Lung Sound Analysis"
    )

    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(totp_uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    qr_code_b64 = base64.b64encode(buffer.getvalue()).decode()

    return render_template("setup_2fa.html", qr_code_b64=qr_code_b64, totp_secret=totp_secret)

@app.route("/verify-2fa", methods=["GET", "POST"])
def verify_2fa():
    """Verification page during login for users with 2FA enabled"""
    pending_user_id = session.get("pending_2fa_user_id")
    if not pending_user_id:
        flash("No pending authentication. Please login first.", "error")
        return redirect(url_for("login"))

    if request.method == "POST":
        verification_code = request.form.get("verification_code")
        user = User.query.get(pending_user_id)

        if not user:
            session.pop("pending_2fa_user_id", None)
            flash("Invalid session. Please login again.", "error")
            return redirect(url_for("login"))

        totp = pyotp.TOTP(user.totp_secret)
        if totp.verify(verification_code):
            session.pop("pending_2fa_user_id", None)
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid verification code. Please try again.", "error")

    return render_template("verify_2fa.html")

@app.route("/disable-2fa", methods=["POST"])
@login_required
def disable_2fa():
    """Disable 2FA for the current user"""
    current_user.is_2fa_enabled = False
    current_user.totp_secret = None
    db.session.commit()
    flash("Two-Factor Authentication has been disabled.", "info")
    return redirect(url_for("settings"))

@app.route("/patients")
@login_required
def patients_list():
    patients = get_clinician_patients()
    return render_template("patients.html", patients=patients)

@app.route("/patient/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    check_patient_access(patient)
    analyses = (
        AnalysisResult.query.join(AudioFile)
        .filter(AudioFile.patient_id == patient_id)
        .order_by(AnalysisResult.analysis_date.desc())
        .all()
    )
    return render_template("patient_detail.html", patient=patient, analyses=analyses)

@app.route("/dashboard")
@login_required
def dashboard():
    if is_admin():
        return redirect(url_for("admin_dashboard"))
    recent_analyses = get_clinician_analyses()[:10]
    return render_template("dashboard.html", recent_analyses=recent_analyses)

@app.route("/upload", methods=["POST"])
@login_required
def upload_audio():
    if is_admin():
        return (
            jsonify({
                "success": False,
                "error": "Access denied",
                "details": "Administrators cannot perform patient diagnoses. Only clinicians can upload and analyze audio files.",
            }),
            403,
        )

    try:
        print("\n" + "=" * 60)
        print("NEW AUDIO UPLOAD REQUEST - HYBRID MODEL")
        print("=" * 60)

        try:
            first_name = request.form.get("first_name")
            middle_name = request.form.get("middle_name", "")
            last_name = request.form.get("last_name")
            patient_id = request.form.get("patient_id")
            date_of_birth = request.form.get("date_of_birth")
            gender = request.form.get("gender")

            if middle_name:
                patient_name = f"{first_name} {middle_name} {last_name}"
            else:
                patient_name = f"{first_name} {last_name}"

            phone_number = request.form.get("phone_number")
            email_address = request.form.get("email_address")
            recording_datetime = request.form.get("recording_datetime")
            visit_type = request.form.get("visit_type")
            recording_location = request.form.get("recording_location", "chest")
            clinical_notes = request.form.get("clinical_notes")

            print(f"[OK] Form data received for hybrid analysis: Patient={patient_name}, ID={patient_id}")
        except Exception as e:
            return (
                jsonify({"success": False, "error": f"Form data error: {str(e)}"}),
                400,
            )

        if not all([first_name, last_name, patient_id]):
            return (
                jsonify({
                    "success": False,
                    "error": "First name, last name, and Patient ID are required",
                }),
                400,
            )

        if not patient_id.isdigit() or len(patient_id) != 4:
            return (
                jsonify({
                    "success": False,
                    "error": "Patient ID must be exactly 4 digits",
                }),
                400,
            )

        if phone_number and (not phone_number.isdigit() or len(phone_number) != 10):
            return (
                jsonify({
                    "success": False,
                    "error": "Phone number must be exactly 10 digits",
                }),
                400,
            )

        if "audio_file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["audio_file"]
        if file.filename == "" or not allowed_file(file.filename):
            return (
                jsonify({
                    "success": False,
                    "error": "Invalid file format. Please upload WAV, MP3, or FLAC",
                }),
                400,
            )

        print(f"[OK] File received for hybrid analysis: {file.filename}")

        try:
            patient = Patient.query.filter_by(patient_id=patient_id, clinician_id=current_user.id).first()
            if not patient:
                dob_parsed = None
                if date_of_birth:
                    try:
                        from datetime import datetime as dt
                        dob_parsed = dt.strptime(date_of_birth, "%Y-%m-%d").date()
                    except ValueError:
                        pass

                patient = Patient(
                    patient_name=patient_name,
                    patient_id=patient_id,
                    date_of_birth=dob_parsed,
                    gender=gender,
                    phone_number=phone_number if phone_number else None,
                    email_address=email_address if email_address else None,
                    clinician_id=current_user.id,
                )
                db.session.add(patient)
                db.session.commit()
                print(f"[OK] New patient created for hybrid analysis: {patient_name} (ID: {patient_id})")
            else:
                print(f"[OK] Existing patient found for hybrid analysis: {patient_name} (ID: {patient_id})")
        except Exception as e:
            db.session.rollback()
            return (
                jsonify({
                    "success": False,
                    "error": f"Database error creating patient: {str(e)}",
                }),
                500,
            )

        try:
            filename = secure_filename(file.filename)
            unique_filename = f"{current_user.id}_{patient_id}_{datetime.now().timestamp()}_{filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            file.save(filepath)
            print(f"[OK] Audio file saved for hybrid analysis: {filepath}")
        except Exception as e:
            return (
                jsonify({"success": False, "error": f"File save error: {str(e)}"}),
                500,
            )

        try:
            recording_dt_parsed = None
            if recording_datetime:
                try:
                    from datetime import datetime as dt
                    recording_dt_parsed = dt.strptime(recording_datetime, "%Y-%m-%dT%H:%M")
                except ValueError:
                    recording_dt_parsed = datetime.now()
            else:
                recording_dt_parsed = datetime.now()

            audio_file = AudioFile(
                filename=filename,
                file_path=filepath,
                recording_location=recording_location,
                recording_datetime=recording_dt_parsed,
                visit_type=visit_type if visit_type else None,
                clinical_notes=clinical_notes if clinical_notes else None,
                patient_id=patient.id,
            )
            db.session.add(audio_file)
            db.session.commit()
            print(f"[OK] Audio file record created in database for hybrid analysis")
        except Exception as e:
            db.session.rollback()
            return (
                jsonify({
                    "success": False,
                    "error": f"Database error saving audio record: {str(e)}",
                }),
                500,
            )

        waveform_filename = f"waveform_{os.path.splitext(unique_filename)[0]}.png"
        spectrogram_filename = f"spectrogram_{os.path.splitext(unique_filename)[0]}.png"

        waveform_path_full = os.path.join(app.config["VISUALIZATIONS_FOLDER"], waveform_filename)
        spectrogram_path_full = os.path.join(app.config["VISUALIZATIONS_FOLDER"], spectrogram_filename)

        os.makedirs(app.config["VISUALIZATIONS_FOLDER"], exist_ok=True)

        waveform_success = generate_waveform_plot(filepath, waveform_path_full)
        spectrogram_success = generate_spectrogram_plot(filepath, spectrogram_path_full)

        waveform_db_path = f"visualizations/{waveform_filename}" if waveform_success else None
        spectrogram_db_path = f"visualizations/{spectrogram_filename}" if spectrogram_success else None

        try:
            print("\nStarting HYBRID disease diagnosis prediction...")
            prediction_result = predict_disease_diagnosis(filepath)

            if not prediction_result.get("success", False):
                error_message = prediction_result.get("error", "Unknown error during hybrid prediction")
                print(f"[ERROR] Hybrid prediction failed: {error_message}")

                try:
                    analysis = AnalysisResult(
                        classification="Error",
                        confidence_score=0.0,
                        disease_diagnosis="Error",
                        disease_confidence=0.0,
                        audio_file_id=audio_file.id,
                        waveform_path=waveform_db_path,
                        spectrogram_path=spectrogram_db_path,
                    )
                    db.session.add(analysis)
                    db.session.commit()
                except:
                    db.session.rollback()

                return (
                    jsonify({
                        "success": False,
                        "error": error_message,
                        "details": "The audio file could not be analyzed by the hybrid model. Please ensure it is a valid respiratory sound recording.",
                    }),
                    500,
                )

        except Exception as e:
            print(f"[ERROR] Exception during hybrid prediction: {str(e)}")
            traceback.print_exc()
            return (
                jsonify({
                    "success": False,
                    "error": "Hybrid prediction error",
                    "details": str(e),
                }),
                500,
            )

        try:
            disease_label = prediction_result.get("disease_label", "Unknown")
            disease_confidence = prediction_result.get("disease_confidence", 0.0)

            print(f"\n[OK] HYBRID prediction successful:")
            print(f"  - Disease: {disease_label} ({disease_confidence:.2f}%)")
            print(f"  - Model: Hybrid (Unsupervised + Supervised)")

            analysis = AnalysisResult(
                classification=disease_label,
                confidence_score=disease_confidence,
                disease_diagnosis=disease_label,
                disease_confidence=disease_confidence,
                audio_file_id=audio_file.id,
                waveform_path=waveform_db_path,
                spectrogram_path=spectrogram_db_path,
            )
            db.session.add(analysis)
            db.session.commit()

            print(f"[OK] Hybrid analysis saved to database (ID: {analysis.id})")
            print("=" * 60 + "\n")

            return jsonify({
                "success": True,
                "analysis_id": analysis.id,
                "disease": disease_label,
                "disease_confidence": round(disease_confidence, 2),
                "all_probabilities": prediction_result.get("all_probabilities", {}),
                "model_type": "Hybrid (Unsupervised + Supervised)",
            })

        except Exception as e:
            db.session.rollback()
            print(f"[ERROR] Error saving hybrid analysis: {str(e)}")
            traceback.print_exc()
            return (
                jsonify({
                    "success": False,
                    "error": "Failed to save hybrid analysis results",
                    "details": str(e),
                }),
                500,
            )

    except Exception as e:
        error_msg = f"Unexpected error in hybrid upload_audio: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        traceback.print_exc()
        print("=" * 60 + "\n")

        try:
            db.session.rollback()
        except:
            pass

        return (
            jsonify({
                "success": False,
                "error": "An unexpected error occurred during hybrid analysis",
                "details": str(e),
            }),
            500,
        )

@app.route("/results/<int:analysis_id>")
@login_required
def view_results(analysis_id):
    analysis = AnalysisResult.query.get_or_404(analysis_id)
    patient = analysis.audio_file.patient
    check_patient_access(patient)

    clinician = patient.clinician

    results_data = {
        "disease_diagnosis": analysis.disease_diagnosis,
        "disease_confidence": analysis.disease_confidence,
        "clinical_interpretation": f"{analysis.disease_diagnosis} Diagnosis (Hybrid Model)",
        "clinical_detail": f"Hybrid respiratory analysis indicates {analysis.disease_diagnosis} with {analysis.disease_confidence:.1f}% confidence using unsupervised and supervised features.",
        "patient_name": patient.patient_name,
        "patient_id": patient.patient_id,
        "age": patient.age,
        "gender": patient.gender,
        "filename": analysis.audio_file.filename,
        "recording_location": analysis.audio_file.recording_location,
        "analysis_date": analysis.analysis_date.strftime("%B %d, %Y at %I:%M %p"),
        "file_size": "3.2 MB",
        "classification": analysis.classification,
        "confidence_score": analysis.confidence_score,
        "waveform_path": analysis.waveform_path,
        "spectrogram_path": analysis.spectrogram_path,
        "attention_map_path": None,
        "clinician_name": clinician.username,
        "clinician_email": clinician.email,
        "clinician_phone": clinician.phone_number if clinician.phone_number else "Not provided",
        "model_type": "Hybrid (Unsupervised + Supervised)",
    }

    return render_template("results.html", **results_data)

@app.route("/reports")
@login_required
def reports():
    if is_admin():
        total_analyses = AnalysisResult.query.count()

        disease_counts = (
            db.session.query(
                AnalysisResult.disease_diagnosis,
                func.count(AnalysisResult.disease_diagnosis),
            )
            .group_by(AnalysisResult.disease_diagnosis)
            .all()
        )

        analyses = AnalysisResult.query.order_by(AnalysisResult.analysis_date.desc()).all()
    else:
        total_analyses = (
            AnalysisResult.query.join(AudioFile)
            .join(Patient)
            .filter(Patient.clinician_id == current_user.id)
            .count()
        )

        disease_counts = (
            db.session.query(
                AnalysisResult.disease_diagnosis,
                func.count(AnalysisResult.disease_diagnosis),
            )
            .join(AudioFile)
            .join(Patient)
            .filter(Patient.clinician_id == current_user.id)
            .group_by(AnalysisResult.disease_diagnosis)
            .all()
        )

        analyses = (
            AnalysisResult.query.join(AudioFile)
            .join(Patient)
            .filter(Patient.clinician_id == current_user.id)
            .order_by(AnalysisResult.analysis_date.desc())
            .all()
        )

    disease_stats = {disease: count for disease, count in disease_counts}

    stats = {
        "total_analyses": total_analyses,
        "normal_count": disease_stats.get("Normal", 0),
        "asthma_count": disease_stats.get("Asthma", 0),
        "copd_count": disease_stats.get("COPD", 0),
        "pneumonia_count": disease_stats.get("Pneumonia", 0),
        "bronchitis_count": disease_stats.get("Bronchitis", 0),
        "bronchiectasis_count": disease_stats.get("Bronchiectasis", 0),
        "urti_count": disease_stats.get("URTI", 0),
    }

    return render_template("reports.html", stats=stats, analyses=analyses)

@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        new_username = request.form.get("username")
        new_email = request.form.get("email")
        new_phone_number = request.form.get("phone_number")
        current_password = request.form.get("current_password")
        new_password = request.form.get("new_password")
        confirm_password = request.form.get("confirm_password")

        changes_made = False

        if new_username and new_username != current_user.username:
            current_user.username = new_username
            changes_made = True

        if new_email and new_email != current_user.email:
            if "admin" in new_email.lower() and current_user.role != "admin":
                flash("Invalid email address. Admin email addresses are restricted.", "error")
                return redirect(url_for("settings"))

            existing_user = User.query.filter_by(email=new_email).first()
            if existing_user and existing_user.id != current_user.id:
                flash("Email already in use by another account.", "error")
                return redirect(url_for("settings"))

            current_user.email = new_email
            changes_made = True

        if new_phone_number and new_phone_number != current_user.phone_number:
            if not new_phone_number.isdigit() or len(new_phone_number) != 10:
                flash("Phone number must be exactly 10 digits.", "error")
                return redirect(url_for("settings"))

            current_user.phone_number = new_phone_number
            changes_made = True

        if new_password:
            if not current_password:
                flash("Current password is required to change password.", "error")
                return redirect(url_for("settings"))

            if not check_password_hash(current_user.password_hash, current_password):
                flash("Current password is incorrect.", "error")
                return redirect(url_for("settings"))

            if new_password != confirm_password:
                flash("New passwords do not match.", "error")
                return redirect(url_for("settings"))

            if len(new_password) < 6:
                flash("New password must be at least 6 characters long.", "error")
                return redirect(url_for("settings"))

            current_user.password_hash = generate_password_hash(new_password)
            changes_made = True

        if changes_made:
            try:
                db.session.commit()
                flash("Settings updated successfully!", "success")
            except Exception as e:
                db.session.rollback()
                flash(f"Error updating settings: {str(e)}", "error")
        else:
            flash("No changes were made.", "info")

        return redirect(url_for("settings"))

    return render_template("settings.html")

@app.route("/clinical-guidelines")
@login_required
def clinical_guidelines():
    """Clinical guidelines page with disease information"""
    return render_template("clinical_guidelines.html")

@app.route("/admin")
@login_required
def admin_dashboard():
    require_admin()
    total_users = User.query.filter_by(role="clinician").count()
    total_patients = Patient.query.count()
    total_analyses = AnalysisResult.query.count()
    recent_analyses = (
        AnalysisResult.query.order_by(AnalysisResult.analysis_date.desc())
        .limit(10)
        .all()
    )
    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_patients=total_patients,
        total_analyses=total_analyses,
        recent_analyses=recent_analyses,
    )

@app.route("/admin/clinicians")
@login_required
def admin_clinicians():
    require_admin()
    clinicians = User.query.filter_by(role="clinician").all()

    clinician_stats = []
    for clinician in clinicians:
        patient_count = Patient.query.filter_by(clinician_id=clinician.id).count()
        analysis_count = (
            AnalysisResult.query.join(AudioFile)
            .join(Patient)
            .filter(Patient.clinician_id == clinician.id)
            .count()
        )
        clinician_stats.append({
            "clinician": clinician,
            "patient_count": patient_count,
            "analysis_count": analysis_count,
        })

    return render_template("admin_clinicians.html", clinician_stats=clinician_stats)

@app.route("/admin/patients")
@login_required
def admin_patients():
    require_admin()
    patients = Patient.query.all()
    clinicians = User.query.filter_by(role="clinician").all()
    return render_template("admin_patients.html", patients=patients, clinicians=clinicians)

@app.route("/admin/reassign-patient", methods=["POST"])
@login_required
def admin_reassign_patient():
    require_admin()
    patient_id = request.form.get("patient_id")
    new_clinician_id = request.form.get("clinician_id")
    patient = Patient.query.get_or_404(patient_id)
    old_clinician_name = patient.clinician.username
    patient.clinician_id = new_clinician_id
    db.session.commit()
    flash(f"Patient reassigned from Dr. {old_clinician_name} to new clinician", "success")
    return redirect(url_for("admin_patients"))

@app.route("/create-admin")
def create_admin():
    secret_key = request.args.get("secret")
    ADMIN_CREATION_SECRET = "LUNG_ADMIN_2024_SECRET_KEY"

    if secret_key != ADMIN_CREATION_SECRET:
        return "Unauthorized: Invalid secret key", 403

    admin = User.query.filter_by(email="admin@lunganalysis.com").first()
    if admin:
        admin.role = "admin"
        db.session.commit()
        return f"User updated to admin role! Email: admin@lunganalysis.com"

    admin = User(
        username="System Admin",
        email="admin@lunganalysis.com",
        password_hash=generate_password_hash("admin123"),
        role="admin",
    )
    db.session.add(admin)
    db.session.commit()
    return "Admin created! Email: admin@lunganalysis.com, Password: admin123 (CHANGE THIS IMMEDIATELY!)"

@app.errorhandler(403)
def forbidden(e):
    flash("Access denied.", "error")
    return redirect(url_for("dashboard"))

@app.errorhandler(404)
def not_found(e):
    flash("Not found.", "error")
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    with app.app_context():
        print("Dropping old tables...")
        db.drop_all()
        print("Creating new tables...")
        db.create_all()
        print("[OK] Database initialized with fresh schema")
        upload_folder = app.config["UPLOAD_FOLDER"]
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        print("[OK] Upload folder ready")

        visualizations_folder = app.config["VISUALIZATIONS_FOLDER"]
        if not os.path.exists(visualizations_folder):
            os.makedirs(visualizations_folder)
        print("[OK] Visualizations folder ready")
    print("\n" + "=" * 50)
    print("Hybrid Lung Sound Analysis System Starting...")
    print("=" * 50)
    print("Open your browser to: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True)