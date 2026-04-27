import librosa
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.mixture import GaussianMixture

def load_audio(file):
    """Loads audio file and returns a mono time series."""
    y, sr = librosa.load(file, sr=None)
    return y, sr

def get_onsets(y, sr):
    """
    Detects high-precision onsets centered on the peak energy.
    Backtracking is disabled to ensure markers align with the visual impulse.
    """
    HOP_LENGTH = 512
    _, y_percussive = librosa.effects.hpss(y)
    
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=HOP_LENGTH)
    
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, 
        sr=sr, 
        hop_length=HOP_LENGTH, 
        backtrack=False
    )
    
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    
    # Extract Peak Amplitudes for Dynamics Analysis
    amplitudes = []
    window_samples = int(0.05 * sr) 
    onset_samples = librosa.time_to_samples(onset_times, sr=sr)
    
    for start_sample in onset_samples:
        end_sample = min(len(y), start_sample + window_samples)
        peak = np.max(np.abs(y[start_sample:end_sample]))
        amplitudes.append(peak)
    
    amplitudes = np.array(amplitudes)
    global_max = np.max(np.abs(y)) if len(y) > 0 else 1.0
    
    return onset_times, onset_env, amplitudes / (global_max + 1e-6)

def detect_global_bpm_filtered(onsets, amplitudes, y, sr, fixed_bpm=None):
    """
    CORE ML ENDPOINT: Determines optimal BPM and Phase.
    If fixed_bpm is provided, skips tempo detection but performs Auto-Sync Phase logic.
    """
    duration = len(y) / sr
    nyquist = 0.5 * sr
    b, a = butter(4, 100 / nyquist, btype='low')
    y_kick = filtfilt(b, a, y)
    
    onset_env = librosa.onset.onset_strength(y=y_kick, sr=sr)
    threshold = 0.4 * np.max(onset_env)
    gated_env = np.where(onset_env > threshold, onset_env, 0)
    
    if fixed_bpm:
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=gated_env, sr=sr, bpm=fixed_bpm)
        candidates = [float(fixed_bpm)]
    else:
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=gated_env, sr=sr)
        float_bpm = float(tempo)
        candidates = [float_bpm, np.floor(float_bpm), np.ceil(float_bpm)]

    best_bpm, max_score, best_offset = candidates[0], -1.0, onsets[0]
    latency_adj = 0.005 # 5ms adjustment for signal propagation/backtracking jitter

    # Competitive Trial Loop: Selects BPM/Phase with highest Gaussian Score
    for cand in candidates:
        if cand <= 40: continue
        
        if len(beat_frames) > 0:
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            offset = (beat_times[0] - latency_adj) % (60.0 / cand)
        else:
            offset = onsets[0] - latency_adj
            
        trial_grid = generate_musical_grid(cand, duration, offset)
        trial_scores, _ = calculate_performance_metrics(onsets, trial_grid, duration)
        mean_score = np.mean(trial_scores)
        
        # Slight bias to snap toward integers if not in fixed mode
        comparison_score = mean_score + 0.5 if (cand.is_integer() and not fixed_bpm) else mean_score

        if comparison_score > max_score:
            max_score, best_bpm, best_offset = comparison_score, cand, offset

    # Octave correction
    if best_bpm > 190: best_bpm /= 2
    
    return round(float(best_bpm), 2), best_offset

def generate_musical_grid(bpm, duration, offset):
    """Creates timestamps for 1/4, 1/8, 1/16, and Triplets."""
    if bpm <= 0: bpm = 120.0 
    beat_duration = 60.0 / bpm
    indices = np.arange(int((duration - offset) / (beat_duration / 4)) + 10)
    
    return {
        "quarter": offset + (indices * beat_duration),
        "eighth": offset + (indices * (beat_duration / 2)),
        "sixteenth": offset + (indices * (beat_duration / 4)),
        "triplet": offset + (indices * (beat_duration / 3))
    }

def calculate_performance_metrics(onsets, grid_dict, duration):
    """Maps hits to nearest grid point and applies Gaussian scoring."""
    valid_points = [grid_dict[k][grid_dict[k] <= duration] for k in grid_dict]
    master_grid = np.unique(np.concatenate(valid_points))
    
    scores, deviations = [], []
    for hit in onsets:
        diffs = np.abs(master_grid - hit)
        delta_t = hit - master_grid[np.argmin(diffs)]
        # Score = 100 * e^(-dt^2 / 2σ^2) | σ = 50ms
        score = 100 * np.exp(-(delta_t**2) / (2 * 0.05**2))
        scores.append(score)
        deviations.append(delta_t)
        
    return np.array(scores), np.array(deviations)

def get_improvement_suggestions(deviations):
    """Generic feedback based on mean deviation."""
    avg_dev = np.mean(deviations)
    tip = "Rushing." if avg_dev < -0.015 else "Dragging." if avg_dev > 0.015 else "Solid clock!"
    return f"{tip} Consistency: {'High' if np.std(deviations) < 0.04 else 'Erratic'}."

def calculate_dynamics_metrics(amplitudes):
    """
    Advanced dynamics analysis using GMM with higher cluster resolution.
    Explores 2 to 4 dynamic layers and selects the best fit.
    """
    if len(amplitudes) < 10: # Higher cluster count needs more data points
        return 0.0, "Need more hits for detailed analysis.", np.zeros(len(amplitudes))

    X = amplitudes.reshape(-1, 1)

    best_gmm = None
    min_bic = np.inf
    for k in range(2, 5): 
        gmm = GaussianMixture(n_components=k, random_state=42).fit(X)
        bic = gmm.bic(X)
        if bic < min_bic:
            min_bic, best_gmm = bic, gmm

    labels = best_gmm.predict(X)
    means = best_gmm.means_.flatten()
    sorted_idx = np.argsort(means)
    rank_map = {old_idx: new_rank for new_rank, old_idx in enumerate(sorted_idx)}
    mapped_labels = np.array([rank_map[l] for l in labels])
    
    num_clusters = len(means)
    cluster_scores = []
    layer_info = []

    def get_layer_name(rank, total):
        if rank == 0: return "Ghost"
        if rank == total - 1: return "Accent"
        if total == 3: return "Regular"
        return f"Layer {rank+1}"

    for rank, idx in enumerate(sorted_idx):
        cluster_data = amplitudes[labels == idx]
        if len(cluster_data) > 1:
            cv = np.std(cluster_data) / (means[idx] + 1e-6)
            score = max(0, 100 * (1 - cv * 3.0))
            cluster_scores.append(score)
            
            name = get_layer_name(rank, num_clusters)
            layer_info.append(f"{name}: {score:.0f}%")

    final_score = np.average(cluster_scores, weights=means[sorted_idx])
    feedback = " | ".join(layer_info)
    
    return round(final_score, 1), f"Found {num_clusters} layers: {feedback}", mapped_labels