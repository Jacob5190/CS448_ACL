```python
import os

readme_content_v2 = """# 🥁 Percussion Practice Analyzer
### Developed by Jacob Wang | UIUC MCS 2026

A high-precision signal processing tool designed to analyze percussion recordings for timing accuracy and dynamic consistency. Developed as part of the CS448 curriculum at the University of Illinois Urbana-Champaign.

## 🚀 Getting Started (Collaborator Setup Guide)

Follow these steps to set up the development environment after pulling the code from GitHub.

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Set Up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies and avoid version conflicts.

**Windows:**
```bash
python -m venv drum_env
.\\\\drum_env\\\\Scripts\\\\activate
```

**macOS / Linux:**
```bash
python3 -m venv drum_env
source drum_env/bin/activate
```

### 3. Install Dependencies
Once the environment is active, install the required signal processing and machine learning libraries:
```bash
pip install --upgrade pip
pip install streamlit librosa matplotlib scikit-learn numpy scipy
```

## 🏗️ Architecture
The system is divided into two primary modules:

1. **`app.py`**: The Streamlit-based frontend. It manages session state, file uploads, and renders all visual data including waveforms, histograms, and dynamic maps.
2. **`audio_processor.py`**: The DSP engine. It handles feature extraction, clock synchronization, and statistical modeling (GMM) of performance data.

## 🧠 Core Logic & Algorithms

### 1. Peak-Centered Onset Detection
Utilizes `librosa.onset.onset_strength` combined with harmonic-percussive separation (HPSS). 
* **Backtrack Disabled**: Markers align with the visual impulse of the strike.
* **Temporal Stability**: Onsets are locked to a fixed 512 `hop_length` to prevent drift in long recordings.

### 2. Competitive BPM Scoring Arena
Evaluates candidates (Float, Floor, and Ceiling) against a shared metronome phase.
* **Unified Anchor**: A 5ms latency adjustment is applied to synchronize the grid with detected transients.
* **Selection Criteria**: The BPM yielding the highest global **Gaussian Timing Score** is selected.

### 3. GMM Dynamics Analysis
Uses **Gaussian Mixture Models (GMM)** to discover natural volume layers.
* **Automated Clustering**: Uses BIC (Bayesian Information Criterion) to determine if a performance has 2 to 5 distinct dynamic layers.
* **Consistency Scoring**: Calculates the Coefficient of Variation (CV) for each discovered layer.

## 🛠️ Teammate Tweak Guide
To experiment with the analysis "feel," adjust these parameters in `audio_processor.py`:

| Variable | Default Value | Impact |
| :--- | :--- | :--- |
| `latency_adj` | `0.005` | Shifts the metronome grid to account for signal propagation. |
| `sigma` | `0.05` | Determines the "strictness" of the Timing Score (Lower = Stricter). |
| `cv` multiplier | `3.0` | Controls sensitivity of the Dynamics Score to volume variations. |
| `threshold` | `0.4` | Filters noise vs. intentional hits for BPM detection. |

---
*Developed by Jacob Wang for CS448 at the University of Illinois Urbana-Champaign.*
"""

with open("README-v2.md", "w", encoding="utf-8") as f:
    f.write(readme_content_v2)

print("Updated README.md with environment setup guide has been generated.")
```

### Key Setup Highlights for Your Teammate:
* **`drum_env` Naming**: The guide uses the environment name found in your project structure for consistency.
* **Platform-Specific Activation**: I’ve included separate activation commands for Windows and Unix-based systems to ensure they can get up and running regardless of their OS.
* **Clean Dependency Path**: The installation command is explicitly listed so they don't have to hunt through the code for imports.

Does this cover the workflow your teammate needs to get their local environment synced with yours?
