# 🫁 Lung Sound Analysis System

**Audio Spectrogram Transformer for Automated Respiratory Disease Detection**

An AI-powered web application for analyzing lung sounds and detecting respiratory diseases using deep learning.

---

## 📋 Overview

This project uses an **Audio Spectrogram Transformer (AST)** to classify lung sounds into four categories:
- Normal
- Crackle
- Wheeze  
- Both (Crackle + Wheeze)

The system includes a Flask web interface for clinicians to upload patient recordings, view analysis results, and manage patient records.

---

## ✨ Features

- 🔐 **User Authentication** - Secure login for clinicians and admins
- 👥 **Patient Management** - Track patient records and analysis history
- 🎵 **Audio Analysis** - Upload and analyze lung sound recordings
- 📊 **Results Dashboard** - View detailed analysis with confidence scores
- 🛡️ **Admin Panel** - User management and system oversight
- 🎨 **Modern UI** - Responsive design with gradient backgrounds

---

## 🛠️ Tech Stack

**Backend**: Flask, SQLAlchemy, Flask-Login  
**Frontend**: HTML5, CSS3, JavaScript  
**ML Model**: PyTorch, Vision Transformer (ViT)  
**Dataset**: ICBHI 2017 Respiratory Sound Database

---

## 🚀 Installation
**1. Clone the repository**
git clone https://github.com/Prospeerous/AST-Lung-Sound-Analysis-and-Diagnosis.git
cd AST-Lung-Sound-Analysis-and-Diagnosis

text

**2. Create virtual environment**
python -m venv venv

text

**3. Activate virtual environment**
Windows
venv\Scripts\activate

macOS/Linux
source venv/bin/activate

text

**4. Install dependencies**
pip install -r requirements.txt

text

**5. Run the application**
python app.py

text

**6. Access the application**

Open your browser and visit: `http://127.0.0.1:5000`

**7. Create admin account (first time only)**

Visit: `http://127.0.0.1:5000/create-admin`

Default credentials:
- Email: `admin@lunganalysis.com`
- Password: `admin123`

---

## 📁 Project Structure

text

---

## 📊 Dataset

**ICBHI 2017 Respiratory Sound Database**
- 920 recordings from 126 patients
- 5.5 hours of audio data
- Recorded at 7 anatomical locations
- Classes: Normal (68.9%), Crackle (10.3%), Wheeze (10.3%), Both (10.3%)

**Preprocessing Pipeline**:
- Resampling to 16kHz
- Audio normalization
- Mel-spectrogram generation (128 mel bins)
- Data augmentation (time stretch, pitch shift, noise injection)

---

## 🧠 Model Architecture

**Audio Spectrogram Transformer (AST)**

- **Base Model**: Vision Transformer (ViT-Base/16) pre-trained on ImageNet
- **Input**: Mel-spectrograms (128×157) resized to 224×224
- **Architecture**: 12 transformer blocks with multi-head attention
- **Parameters**: 86.2M trainable parameters
- **Output**: 4-class classification with confidence scores

**Training Configuration**:
- Optimizer: AdamW (lr=1e-4, weight decay=0.01)
- Loss: Weighted Cross-Entropy
- Scheduler: Cosine Annealing (30 epochs)
- Batch Size: 32
- Hardware: Kaggle P100 GPU

---

## 🚧 Project Status

**Development Branch** - Web interface complete, AST model training in progress.

### Completed ✅
- Flask web application architecture
- User authentication & authorization system
- Patient database management
- Audio file upload and storage
- Results display interface
- Admin panel with user management
- Responsive UI with modern design
- Reports and settings pages

### In Progress 🔄
- AST model training on Kaggle
- Model integration into Flask app
- Audio preprocessing optimization
- Performance tuning

### Planned 📅
- Spectrogram visualization
- Attention map display
- Advanced reporting features
- Batch processing capabilities
- RESTful API

---

## 📖 Usage

### For Clinicians

1. **Register** - Create account with email and password
2. **Login** - Access your dashboard
3. **Upload Recording**:
   - Enter patient information
   - Select recording location
   - Upload audio file (WAV/MP3/FLAC)
4. **View Results** - See classification and confidence score
5. **Manage Patients** - Track patient history and analyses

### For Administrators

1. **Login** with admin credentials
2. **Access Admin Panel** - View system statistics
3. **Manage Patients** - Reassign patients to different clinicians
4. **Monitor Activity** - Track system usage

---

## 👥 Team

**ICS 4B Academic Project**

- **Abigael Wambui Mwangi**
- **Stanslaus Mwongela**

**Institution**: [Strathmore University]

---

## 📄 License

This project is for academic purposes only. Not for commercial or clinical use without proper validation.

---

## 🙏 Acknowledgments

- ICBHI for providing the respiratory sound database
- PyTorch and Flask communities for excellent frameworks
- Kaggle for free GPU resources
- My academic supervisor for guidance

---

## 🔗 Links

- [ICBHI Dataset](https://bhichallenge.med.auth.gr/)
- [GitHub Repository](https://github.com/Prospeerous/AST-Lung-Sound-Analysis-and-Diagnosis)
- [Project Documentation](your-docs-link)

---

<div align="center">

**Building the future of respiratory disease detection**

Made with ❤️ by Team Prospeerous

</div>
