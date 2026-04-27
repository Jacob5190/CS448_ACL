### Set Up a Virtual Environment
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

### Install Dependencies
Once the environment is active, install the required signal processing and machine learning libraries:
```bash
pip install --upgrade pip
pip install streamlit librosa matplotlib scikit-learn numpy scipy
```

## Architecture
The system is divided into two primary modules:

1. **`app.py`**: The Streamlit-based frontend. It manages session state, file uploads, and renders all visual data including waveforms, histograms, and dynamic maps.
2. **`audio_processor.py`**: The DSP engine. It handles feature extraction, clock synchronization, and statistical modeling (GMM) of performance data.

## Core Logic & Algorithms

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
* **Consistency Scoring**: Calculates the Coefficient of Variation for each discovered layer.

## Tweak Guide
To experiment with the analysis "feel," adjust these parameters in `audio_processor.py`:

| Variable | Default Value | Impact |
| :--- | :--- | :--- |
| `latency_adj` | `0.005` | Shifts the metronome grid to account for signal propagation. |
| `sigma` | `0.05` | Determines the "strictness" of the Timing Score (Lower = Stricter). |
| `cv` multiplier | `3.0` | Controls sensitivity of the Dynamics Score to volume variations. |
| `threshold` | `0.4` | Filters noise vs. intentional hits for BPM detection. |
