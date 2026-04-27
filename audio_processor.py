import librosa
import numpy as np
from scipy.signal import butter, filtfilt
from scipy import signal, stats
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
    
    # Extract Peak Amplitudes and Decay Rates for Dynamics Analysis
    amplitudes = []
    decay_rates = []
    window_samples = int(0.15 * sr) # 150ms window
    onset_samples = librosa.time_to_samples(onset_times, sr=sr)
    envelope = np.abs(signal.hilbert(y))
    
    for i, start_sample in enumerate(onset_samples):
        end_sample = min(len(y), start_sample + window_samples)
        if i < len(onset_samples) - 1:
            end_sample = min(end_sample, onset_samples[i+1])
            
        strike_data = envelope[start_sample:end_sample]
        if len(strike_data) < 10:
            amplitudes.append(0.0)
            decay_rates.append(0.0)
            continue
            
        peak = np.max(strike_data)
        amplitudes.append(peak)
        
        peak_idx = np.argmax(strike_data)
        decay_data = strike_data[peak_idx:]
        if len(decay_data) > 5:
            decay_data = np.maximum(decay_data, 1e-6)
            log_decay = np.log(decay_data)
            x = np.arange(len(log_decay)) / sr
            slope, _, _, _, _ = stats.linregress(x, log_decay)
            decay_rates.append(-slope) # Positive decay rate
        else:
            decay_rates.append(0.0)
    
    amplitudes = np.array(amplitudes)
    decay_rates = np.array(decay_rates)
    global_max = np.max(np.abs(y)) if len(y) > 0 else 1.0
    
    return onset_times, onset_env, amplitudes / (global_max + 1e-6), decay_rates

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
        trial_scores, _, _, _, _ = calculate_performance_metrics(onsets, trial_grid, duration)
        mean_score = np.mean(trial_scores) if len(trial_scores) > 0 else 0
        
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

def align_onsets_dtw(ref_times, prac_times, max_deviation=0.1):
    """Sequence alignment using Dynamic Time Warping."""
    n, m = len(ref_times), len(prac_times)
    if n == 0 or m == 0: return [], 0.0
    
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i, j] = abs(ref_times[i] - prac_times[j])
            
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = cost[i-1, j-1]
            dtw[i, j] = c + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
            
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        choices = [dtw[i-1, j-1], dtw[i-1, j], dtw[i, j-1]]
        best = np.argmin(choices)
        if best == 0:
            i, j = i-1, j-1
        elif best == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    
    aligned_pairs = []
    for r_idx, p_idx in path:
        if abs(ref_times[r_idx] - prac_times[p_idx]) <= max_deviation:
            aligned_pairs.append((r_idx, p_idx))
            
    return aligned_pairs, dtw[n, m]

def calculate_performance_metrics(onsets, grid_dict, duration, max_deviation=0.1):
    """Maps hits to nearest grid point using DTW and applies Gaussian scoring."""
    valid_points = [grid_dict[k][grid_dict[k] <= duration] for k in grid_dict]
    master_grid = np.unique(np.concatenate(valid_points))
    
    aligned_pairs, _ = align_onsets_dtw(master_grid, onsets, max_deviation)
    
    final_matches = []
    used_gt, used_prac = set(), set()
    sorted_pairs = sorted(aligned_pairs, key=lambda p: abs(master_grid[p[0]] - onsets[p[1]]))
    
    for gt_idx, prac_idx in sorted_pairs:
        if gt_idx not in used_gt and prac_idx not in used_prac:
            final_matches.append((gt_idx, prac_idx))
            used_gt.add(gt_idx)
            used_prac.add(prac_idx)
            
    final_matches.sort()
    
    scores, deviations = [], []
    for gt_idx, prac_idx in final_matches:
        delta_t = onsets[prac_idx] - master_grid[gt_idx]
        # Score = 100 * e^(-dt^2 / 2σ^2) | σ = 50ms
        score = 100 * np.exp(-(delta_t**2) / (2 * 0.05**2))
        scores.append(score)
        deviations.append(delta_t)
        
    missed_indices = [i for i in range(len(master_grid)) if i not in used_gt]
    extra_indices = [i for i in range(len(onsets)) if i not in used_prac]
        
    return np.array(scores), np.array(deviations), len(missed_indices), len(extra_indices), master_grid

def get_improvement_suggestions(deviations, num_missed=0, num_extra=0):
    """Generic feedback based on mean deviation and hit counts."""
    if len(deviations) == 0:
        return "No matched hits found."
    avg_dev = np.mean(deviations)
    tip = "Rushing." if avg_dev < -0.015 else "Dragging." if avg_dev > 0.015 else "Solid clock!"
    consistency = "High" if np.std(deviations) < 0.04 else "Erratic"
    
    feedback = f"{tip} Consistency: {consistency}."
    if num_missed > 0 or num_extra > 0:
        feedback += f" (Missed: {num_missed}, Extra: {num_extra})"
    return feedback

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