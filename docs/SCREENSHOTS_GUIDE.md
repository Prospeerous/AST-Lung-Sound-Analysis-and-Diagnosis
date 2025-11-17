# Screenshots Guide

This guide explains what screenshots to capture and add to your repository to make it more appealing.

## Why Screenshots Matter

Screenshots help potential users, employers, and lecturers:
- Quickly understand what your application does
- See the quality of your UI/UX design
- Visualize the workflow and features
- Assess the professionalism of the project

## Recommended Screenshots

### 1. Landing/Home Page
**Location**: `static/screenshots/01_landing_page.png`

**What to capture:**
- Clean, welcoming home page
- Clear value proposition
- Navigation elements

**Best practices:**
- Use a clean, professional browser window
- Full page screenshot or above-the-fold section
- Make sure no personal/test data is visible

---

### 2. Login Page
**Location**: `static/screenshots/02_login_page.png`

**What to capture:**
- Login form
- 2FA option (if visible)
- Professional design

---

### 3. Dashboard
**Location**: `static/screenshots/03_dashboard.png`

**What to capture:**
- Clean dashboard layout
- Statistics/metrics visible
- Recent analyses
- Navigation menu

**Important:**
- Use sample/demo data (not real patient data)
- Show meaningful statistics (not all zeros)
- Professional appearance

---

### 4. Analysis Upload Page
**Location**: `static/screenshots/04_upload_page.png`

**What to capture:**
- Upload form with all fields visible
- Clear instructions
- Professional layout

---

### 5. Analysis Results
**Location**: `static/screenshots/05_results_page.png`

**What to capture:**
- Diagnosis results prominently displayed
- Confidence scores
- Visualizations (waveform and spectrogram)
- Patient information (anonymized)
- Professional color coding

**Best practices:**
- Use an interesting sample case (not just "Normal")
- Make sure visualizations are clear and readable
- Show high confidence scores for impact

---

### 6. Patient Management
**Location**: `static/screenshots/06_patients_page.png`

**What to capture:**
- List of patients (use fake/demo data)
- Clean table layout
- Search/filter options if available

---

### 7. Visualizations Close-up
**Location**: `static/screenshots/07_visualizations.png`

**What to capture:**
- High-quality waveform plot
- High-quality mel spectrogram
- Clear labels and axes

**Best practices:**
- Capture an interesting audio sample (with clear patterns)
- Ensure good contrast and readability
- Professional color scheme

---

### 8. 2FA Setup (Optional but impressive)
**Location**: `static/screenshots/08_2fa_setup.png`

**What to capture:**
- QR code setup page
- Clean security features
- Professional security UI

---

### 9. Admin Dashboard (Optional)
**Location**: `static/screenshots/09_admin_dashboard.png`

**What to capture:**
- Admin-specific features
- System-wide statistics
- User management interface

---

### 10. Mobile Responsive View (Bonus)
**Location**: `static/screenshots/10_mobile_view.png`

**What to capture:**
- How the app looks on mobile devices
- Responsive design elements

---

## How to Capture Screenshots

### Tools for High-Quality Screenshots

**Windows:**
- Snipping Tool (Built-in)
- Greenshot (Free)
- ShareX (Free, open-source)

**macOS:**
- Command + Shift + 4 (Built-in)
- CleanShot X (Paid, professional)

**Cross-platform:**
- Browser DevTools (F12) → Toggle device toolbar
- Awesome Screenshot (Browser extension)
- Lightshot (Free)

### Screenshot Best Practices

1. **Use a clean browser**
   - Clear browser extensions from view
   - Use a professional browser (Chrome, Firefox, Edge)
   - Zoom level: 100% (sometimes 90% for more content)

2. **Prepare demo data**
   - Create sample patients with realistic but fake names
   - Use dates that make sense
   - Generate multiple analyses to show activity

3. **Image quality**
   - Use PNG format (not JPG) for crisp text
   - Minimum width: 1200px (for clarity on GitHub)
   - Keep file sizes reasonable (<500KB per image)

4. **Privacy & Security**
   - NEVER use real patient data
   - Hide any real email addresses
   - Use demo/test credentials
   - No sensitive configuration visible

5. **Consistency**
   - Use the same browser window size for all screenshots
   - Keep the same zoom level
   - Maintain consistent branding/theme

## Annotating Screenshots (Optional but Helpful)

Add arrows, highlights, or labels to emphasize key features:

**Tools:**
- Microsoft PowerPoint (Add shapes and text)
- GIMP (Free, open-source)
- Photoshop (Professional)
- Online: Figma, Canva

**What to highlight:**
- Key features or innovations
- Important UI elements
- Workflow steps

## Adding Screenshots to README

Once you have screenshots, update your README:

```markdown
## Screenshots

### Dashboard
![Dashboard](static/screenshots/03_dashboard.png)
*Clinician dashboard showing patient statistics and recent analyses*

### Analysis Results
![Results](static/screenshots/05_results_page.png)
*Comprehensive analysis results with AI diagnosis, confidence scores, and visualizations*

### Visualizations
![Visualizations](static/screenshots/07_visualizations.png)
*Waveform and mel spectrogram generated from lung sound recordings*
```

## Creating a Demo GIF (Advanced)

Animated GIFs can showcase your workflow:

**Tools:**
- ScreenToGif (Windows, free)
- LICEcap (Cross-platform, free)
- Gifox (macOS)

**What to show:**
- Complete workflow: Upload → Analysis → Results (15-30 seconds)
- Keep it concise and smooth
- Add to README header for maximum impact

## File Organization

```
static/
└── screenshots/
    ├── 01_landing_page.png
    ├── 02_login_page.png
    ├── 03_dashboard.png
    ├── 04_upload_page.png
    ├── 05_results_page.png
    ├── 06_patients_page.png
    ├── 07_visualizations.png
    ├── 08_2fa_setup.png
    ├── 09_admin_dashboard.png
    ├── 10_mobile_view.png
    └── demo.gif (optional)
```

## Quick Checklist

Before adding screenshots to your repository:

- [ ] All screenshots use demo/fake data only
- [ ] No real patient information visible
- [ ] No sensitive configuration or secrets visible
- [ ] Images are PNG format for clarity
- [ ] File sizes are reasonable (<500KB each)
- [ ] All screenshots are professional quality
- [ ] Consistent browser window size
- [ ] Good lighting/contrast in screenshots
- [ ] Updated README.md with screenshot embeds
- [ ] Added descriptive captions for each screenshot

## Example README Section

```markdown
## Application Screenshots

<p align="center">
  <img src="static/screenshots/03_dashboard.png" alt="Dashboard" width="800"/>
  <br/>
  <em>Clinician Dashboard</em>
</p>

<p align="center">
  <img src="static/screenshots/05_results_page.png" alt="Results" width="800"/>
  <br/>
  <em>Analysis Results with Visualizations</em>
</p>

<p align="center">
  <img src="static/screenshots/07_visualizations.png" alt="Visualizations" width="800"/>
  <br/>
  <em>Audio Waveform and Mel Spectrogram</em>
</p>
```

## Need Help?

For questions about screenshots:
- Check GitHub's guide on [adding images to README](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#images)
- Look at other successful medical/healthcare projects on GitHub for inspiration

---

**Remember**: Quality screenshots can significantly increase the appeal and credibility of your project!
