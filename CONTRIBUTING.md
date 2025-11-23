# Contributing to Hybrid Lung Sound Analysis System

Thank you for your interest in contributing to this project! We welcome contributions from the community to help improve and enhance this medical diagnostic system.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a code of conduct that promotes a respectful and inclusive environment. By participating, you are expected to uphold this code.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community and project
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include as many details as possible:

**Bug Report Template:**
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. Upload file '...'
4. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g., Windows 11, macOS 14, Ubuntu 22.04]
 - Python Version: [e.g., 3.9.7]
 - Browser (if web-related): [e.g., Chrome 120, Firefox 121]

**Additional context**
Any other relevant information.
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear title and description** of the suggested enhancement
- **Use cases** explaining why this would be useful
- **Examples** of how the feature would work
- **Mockups or diagrams** if applicable

### Code Contributions

We actively welcome your pull requests:

1. Fork the repository and create your branch from `main`
2. If you've added code that should be tested, add tests
3. Ensure your code follows the style guidelines
4. Make sure your code passes all tests
5. Update documentation as needed
6. Issue that pull request!

## Development Setup

### Prerequisites
- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### Setup Steps

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AST-Lung-Sound-Analysis-and-Diagnosis.git
   cd AST-Lung-Sound-Analysis-and-Diagnosis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**
   ```bash
   python app.py  # Will create initial database on first run
   ```

5. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

**General Guidelines:**
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters (120 for long URLs/paths)
- Use descriptive variable names (avoid single letters except in loops)
- Add docstrings to all functions, classes, and modules

**Naming Conventions:**
- `snake_case` for functions and variables
- `PascalCase` for class names
- `UPPER_CASE` for constants
- Prefix private methods/variables with underscore: `_private_method`

**Example:**
```python
class AudioProcessor:
    """Handles audio file preprocessing and feature extraction.

    This class provides methods for loading, resampling, and
    converting audio files to mel spectrograms.
    """

    MAX_AUDIO_LENGTH = 4  # seconds

    def __init__(self, sample_rate=16000):
        """Initialize the audio processor.

        Args:
            sample_rate (int): Target sample rate in Hz
        """
        self.sample_rate = sample_rate

    def process_audio(self, file_path):
        """Process an audio file and extract features.

        Args:
            file_path (str): Path to audio file

        Returns:
            np.ndarray: Mel spectrogram of shape (128, 224)

        Raises:
            ValueError: If file format is not supported
        """
        # Implementation here
        pass
```

### HTML/CSS Guidelines
- Use semantic HTML5 elements
- Keep CSS organized and commented
- Use Bootstrap classes consistently
- Ensure accessibility (alt text, ARIA labels)

### Commit Messages

Write clear, concise commit messages:

**Format:**
```
type(scope): brief description

Detailed explanation if needed.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic changes)
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(auth): add two-factor authentication support

Implement TOTP-based 2FA with QR code generation.
Users can now enable 2FA from settings page.

Fixes #45

---

fix(analysis): handle missing audio metadata gracefully

Previously crashed when audio files lacked duration metadata.
Now defaults to processing full file length.

Fixes #67
```

## Pull Request Process

1. **Update Documentation**
   - Update README.md if you've added features
   - Add docstrings to new functions/classes
   - Update CHANGELOG.md

2. **Test Your Changes**
   - Run the application and test manually
   - Test edge cases and error conditions
   - Verify existing functionality still works

3. **Submit Pull Request**
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changes you made and why
   - Include screenshots for UI changes

4. **Code Review**
   - Be open to feedback and suggestions
   - Make requested changes promptly
   - Ask questions if something is unclear

5. **Merge Requirements**
   - At least one approval from maintainers
   - All discussions resolved
   - No merge conflicts
   - Passes all checks (when CI is set up)

## Project Structure

Understanding the codebase structure:

```
AST-Lung-Sound-Analysis-and-Diagnosis/
├── app.py                  # Main Flask application and model
├── database/
│   └── models.py          # SQLAlchemy database models
├── templates/             # HTML templates
│   ├── base.html         # Base template with navbar
│   ├── dashboard.html    # Main dashboard
│   └── ...               # Other page templates
├── static/
│   ├── css/             # Stylesheets
│   ├── uploads/         # User-uploaded audio files
│   └── visualizations/  # Generated plots
└── outputs/
    └── hybrid_model.pth  # Trained model weights
```

## Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- [ ] Automated testing (unit tests, integration tests)
- [ ] Performance optimization for model inference
- [ ] Additional audio preprocessing techniques
- [ ] Mobile-responsive UI improvements
- [ ] Accessibility enhancements

### Medium Priority
- [ ] Export functionality (PDF reports)
- [ ] Advanced analytics and visualizations
- [ ] Multi-language support
- [ ] Dark mode theme
- [ ] REST API for programmatic access

### Research & Advanced
- [ ] Model architecture improvements
- [ ] Additional disease categories
- [ ] Explainable AI (model interpretability)
- [ ] Real-time audio analysis
- [ ] Integration with medical devices

## Questions?

If you have questions about contributing:

1. Check the [README.md](README.md) for basic information
2. Look through existing [Issues](https://github.com/Prospeerous/AST-Lung-Sound-Analysis-and-Diagnosis/issues)
3. Open a new issue with the `question` label
4. Contact: abigael.mwangi@strathmore.edu

## Recognition

Contributors will be recognized in:
- README.md acknowledgments section
- Release notes for significant contributions
- Project documentation

Thank you for helping make this project better!

---

**Note**: This project is for research and educational purposes. Medical decisions should always be made by qualified healthcare professionals.
