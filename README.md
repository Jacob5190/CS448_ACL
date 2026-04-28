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

### Start the Program
```bash
streamlit run app.py
```

## Architecture
The system is divided into two primary modules:

1. **`app.py`**: The Streamlit-based frontend. Manages session state, file uploads, and renders all visualizations — waveforms, timing histograms, dynamic maps, and the interactive WaveSurfer player.
2. **`audio_processor.py`**: The DSP and ML e   ngine. Handles onset detection, BPM estimation, master grid inference, DTW alignment, and GMM-based dynamics modeling.

---

## Core Logic & Algorithms

### 1. Peak-Centered Onset Detection
Utilizes `librosa.onset.onset_strength` combined with harmonic-percussive separation (HPSS).
* **Backtrack Disabled**: Markers align with the visual impulse of the strike.
* **Temporal Stability**: Onsets are locked to a fixed 512 `hop_length` to prevent drift in long recordings.
* **Per-Hit Features**: For each detected onset, a 150ms analysis window extracts the **peak amplitude** (via Hilbert envelope) and **decay rate** (via log-linear regression on the envelope tail). These feed into dynamics scoring and technique analysis respectively.

### 2. Competitive BPM Scoring Arena
Evaluates BPM candidates (float, floor, and ceiling) against a shared metronome phase.
* **Kick-Frequency Gating**: Audio is low-pass filtered to 100 Hz before beat tracking, isolating the kick drum and reducing false positives from cymbals.
* **Unified Anchor**: A 5ms latency adjustment is applied to synchronize the grid with detected transients.
* **Selection Criteria**: The BPM yielding the highest mean **Gaussian Timing Score** across all onsets is selected.

### 3. Master Grid Inference (Pattern Discovery)
Rather than scoring hits against a rigid metronome, the system infers the **intended rhythmic pattern** directly from the performer's playing and uses that as the reference. This allows the analyzer to distinguish mistakes from deliberate stylistic choices. The pipeline has four stages:

#### Stage 1 — Quantize to Sixteenth-Note Grid
Every detected onset is snapped to its nearest sixteenth-note grid point (within a 120ms tolerance window), producing a binary sequence where `1` = a hit was played on that slot and `0` = silence.

#### Stage 2 — Structural Segmentation
The binary sequence is divided into overlapping windows for independent pattern analysis:
- Recordings under 128 grid steps → analyzed as one block
- 128–256 steps → 128-step windows
- Over 256 steps → 256-step windows

This allows the analyzer to handle recordings where a drummer switches patterns mid-performance (e.g., verse → chorus).

#### Stage 3 — Autocorrelation Pattern Length Detection (`_detect_pattern_length`)
For each segment, the most likely **musical pattern length** (in sixteenth notes) is determined by computing the **autocorrelation function (ACF)** of the binary hit sequence. Only musically meaningful lags are evaluated: `[8, 12, 16, 24, 32, 48, 64]` sixteenth notes (corresponding to 2-bar, 3-beat, 4-bar patterns, etc.). A small length penalty (`1 - lag × 0.002`) biases toward shorter, simpler patterns when scores are close.

#### Stage 4 — Pattern Folding & Consensus (`_infer_intended_pattern`)
The binary sequence is "folded" over the detected pattern length — like stacking sheets of paper — and each sixteenth-note position is counted across all folds. Positions that were played in **≥ 40% of repetitions** are considered intentional hits and form the **intended pattern template**. This threshold naturally absorbs occasional missed or added notes as performance variation rather than pattern changes.

The template is then unrolled back across the full timeline to produce the **Master Grid**: a set of reference timestamps representing what the performer *meant* to play.

### 4. DTW Alignment & Gaussian Scoring
Hits are matched to master grid points using **Dynamic Time Warping (DTW)**, which handles flexible many-to-one alignment across the full sequence. After DTW produces candidate pairs, a **greedy bipartite matching** step resolves conflicts — each hit and each grid point is used at most once, with the best-fitting (lowest deviation) matches given priority.

Each matched hit receives a **Gaussian timing score**:

```
score = 100 × exp(−Δt² / (2 × 0.05²))
```

where `Δt` is the timing deviation in seconds (σ = 50ms). Unmatched grid points are counted as **Missed Hits**. Unmatched onsets are classified as either **Ghost Notes** (if their GMM dynamic layer is the lowest amplitude cluster) or **Flubs** (unexpected hits at full volume).

### 5. GMM Dynamics Analysis (Refactored)
Uses **Gaussian Mixture Models** to discover natural volume layers in the performance. The internal implementation has been decomposed into focused helpers:
* **`_fit_best_gmm`**: Fits k=2..4 component GMMs and selects the model with the lowest **BIC (Bayesian Information Criterion)**.
* **`_calculate_layer_scores`**: For each discovered layer, computes the **Coefficient of Variation (CV)** of amplitudes within that cluster. Lower CV = more consistent velocity control = higher score. Only layers with more than one hit contribute to the weighted average final score, preventing single-sample clusters from skewing the result.
* **Layer Naming**: Clusters are ranked by mean amplitude. The softest is labeled `Ghost`, the loudest is `Accent`, and any middle layers are named `Regular` or `Layer N`.

---

## Dashboard Metrics

| Metric | Description |
| :--- | :--- |
| **Timing Score** | Mean Gaussian score (0–100) across all matched hits |
| **Dynamics Score** | Weighted average CV-based consistency score across GMM layers |
| **Analyzed BPM** | Detected or user-supplied tempo |
| **Missed Hits** | Grid points in the Master Grid with no matching onset |
| **Ghost Notes** | Extra onsets classified as the softest dynamic layer |
| **Flubs** | Extra onsets at normal/loud volume with no grid match |

---

## Visualizations

- **Timing Tendency Profile**: Histogram of timing deviations (ms), colored blue for early/rushing and red for late/dragging.
- **Dynamic Consistency Map (GMM)**: Bar chart of strike intensities colored by discovered layer, with a legend showing Ghost/Regular/Accent layers.
- **Decay Rate Profile**: Relative sustain envelope per hit — useful for detecting inconsistent stick grip or rimshots.
- **Hit-by-Hit Timeline**: Multi-row waveform view with color-coded onset markers (🟢 >80, 🟠 >50, 🔴 error, 🔵 ghost) and the Master Grid overlaid as blue reference lines.
- **Interactive WaveSurfer Player**: An embedded WaveSurfer.js waveform player with the full Master Grid rendered as millisecond-precision markers. Supports zoom, playback controls, and region inspection.

---

## Tweak Guide

| Variable | Default | Impact |
| :--- | :--- | :--- |
| `latency_adj` | `0.005` | Shifts the metronome grid to account for signal propagation delay (seconds). |
| `sigma` | `0.05` | Controls strictness of the Gaussian timing score. Lower = stricter. |
| `cv` multiplier | `3.0` | Sensitivity of the Dynamics Score to intra-layer velocity variation. |
| `threshold` (BPM) | `0.4` | Kick envelope noise gate for beat tracking. |
| `tolerance` (quantize) | `0.12` | Max distance (seconds) to snap an onset to a sixteenth-note grid slot. |
| `fold_threshold` | `0.4` | Fraction of folds a position must appear in to be included in the Master Grid. |
| `max_deviation` (DTW) | `0.1` | Maximum allowed timing error (seconds) for a hit to be considered a match. |
