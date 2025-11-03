# Audio Spectrogram Transformer for Lung Sound Classification

## Project Overview

This project implements a deep learning model based on the Audio Spectrogram Transformer (AST) architecture for automatic classification of lung sounds to support respiratory disease diagnosis. The goal is to enhance early detection and improve diagnostic accuracy for conditions like asthma, COPD, pneumonia, and bronchitis using audio recordings of lung sounds.

The model is trained and evaluated on publicly available respiratory sound datasets including ICBHI 2017, KAUST Respiratory, and SPRSound datasets. Data preprocessing includes denoising, resampling, segmenting, and conversion to mel-spectrograms, making it suitable for transformer-based analysis.

## Features

- Automated respiratory sound classification using AST deep learning architecture
- Processing of diverse, real-world audio data with noise reduction and augmentation
- Multi-class classification of common respiratory conditions
- User-friendly web interface (Flask-based) for uploading audio and viewing diagnostic results
- Model training and evaluation with cross-validation for robustness
- History tracking of analysis requests for transparency and user feedback

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- PyTorch, TensorFlow, and Keras
- Audio processing libraries: librosa, soundfile, noisereduce
- Flask web framework
- Additional libraries as listed in `requirements.txt`

### Setup Instructions

1. Clone the repository
2. Install required libraries using:
    ```
    pip install -r requirements.txt
    ```
3. Download respiratory sound datasets (ICBHI, KAUST, SPRSound) and place them in the `data` directory
4. Run the data preprocessing script to generate spectrograms
5. Train the model using the provided training notebook or script
6. Launch the Flask app to start the web interface for audio classification

## Usage

Use the Flask web app to upload lung sound audio files (.wav). The system preprocesses the audio, applies the trained AST model, and displays the predicted respiratory condition. Users can also view analysis history and track model performance.

For model training and evaluation, refer to the Jupyter notebook `ast-2-model-pipeline.ipynb` which details environment setup, data loading, augmentation, model definition, training loop, and evaluation metrics.

## Model Performance

- Achieves high accuracy and F1-scores on multi-class classification tasks
- Robust against noisy and diverse real-world respiratory sound recordings
- Utilizes stratified cross-validation and weighted loss functions to handle class imbalance

## Project Structure
- data/ # Raw and preprocessed respiratory sound datasets
- notebooks/ # Training, evaluation, and preprocessing notebooks
- app/ # Flask web application source code
- models/ # Saved model checkpoints
- docs/ # Project documentation and design diagrams
- requirements.txt # Project dependencies

## Contributing

Contributions are welcome! Please open issues to report bugs or suggest features. Fork the repository and submit pull requests with descriptive commit messages. Ensure code follows the existing style and passes tests.

## License

This project is licensed under the MIT License.

## Contact

Abigael Wambui Mwangi  
Email:abigael.mwangi@strathmore.edu  
GitHub:Prospeerous



