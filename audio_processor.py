import librosa
import numpy as np
from scipy.signal import butter, filtfilt
from scipy import signal, stats
from sklearn.mixture import GaussianMixture

def load_audio(file):
    """Loads audio file and returns a mono time series."""
    y, sr = librosa.load(file, sr=None)
    return y, sr

def _calculate_onset_features(y, onset_times, sr):
    """Calculates peak amplitudes and decay rates for each onset."""
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
            
    return np.array(amplitudes), np.array(decay_rates)

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
    
    amplitudes, decay_rates = _calculate_onset_features(y, onset_times, sr)
    
    global_max = np.max(np.abs(y)) if len(y) > 0 else 1.0
    
    return onset_times, onset_env, amplitudes / (global_max + 1e-6), decay_rates

def _prepare_kick_env(y, sr):
    """Filters audio to focus on kick frequencies and returns onset strength envelope."""
    nyquist = 0.5 * sr
    b, a = butter(4, 100 / nyquist, btype='low')
    y_kick = filtfilt(b, a, y)
    
    onset_env = librosa.onset.onset_strength(y=y_kick, sr=sr)
    threshold = 0.4 * np.max(onset_env)
    return np.where(onset_env > threshold, onset_env, 0)

def _evaluate_bpm_candidates(onsets, candidates, duration, sr, beat_frames, fixed_bpm):
    """Selects the best BPM candidate based on musical grid alignment scores."""
    best_bpm, max_score, best_offset = candidates[0], -1.0, onsets[0]
    latency_adj = 0.005 # 5ms adjustment
    
    for cand in candidates:
        if cand <= 40: continue
        
        if len(beat_frames) > 0:
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            offset = (beat_times[0] - latency_adj) % (60.0 / cand)
        else:
            offset = onsets[0] - latency_adj
            
        trial_grid = generate_musical_grid(cand, duration, offset)
        trial_scores = calculate_simple_metrics(onsets, trial_grid, duration)
        mean_score = np.mean(trial_scores) if len(trial_scores) > 0 else 0
        
        # Slight bias to snap toward integers if not in fixed mode
        comparison_score = mean_score + 0.5 if (cand.is_integer() and not fixed_bpm) else mean_score

        if comparison_score > max_score:
            max_score, best_bpm, best_offset = comparison_score, cand, offset
            
    return best_bpm, best_offset

def detect_global_bpm_filtered(onsets, amplitudes, y, sr, fixed_bpm=None):
    """
    CORE ML ENDPOINT: Determines optimal BPM and Phase.
    If fixed_bpm is provided, skips tempo detection but performs Auto-Sync Phase logic.
    """
    duration = len(y) / sr
    gated_env = _prepare_kick_env(y, sr)
    
    if fixed_bpm:
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=gated_env, sr=sr, bpm=fixed_bpm)
        candidates = [float(fixed_bpm)]
    else:
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=gated_env, sr=sr)
        float_bpm = float(tempo[0])
        candidates = [float_bpm, np.floor(float_bpm), np.ceil(float_bpm)]

    best_bpm, best_offset = _evaluate_bpm_candidates(onsets, candidates, duration, sr, beat_frames, fixed_bpm)

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

def calculate_simple_metrics(onsets, grid_dict, duration):
    """Basic nearest-neighbor scoring used purely for BPM estimation."""
    valid_points = [grid_dict[k][grid_dict[k] <= duration] for k in grid_dict]
    if not valid_points: return np.array([])
    master_grid = np.unique(np.concatenate(valid_points))
    
    scores = []
    for hit in onsets:
        diffs = np.abs(master_grid - hit)
        delta_t = hit - master_grid[np.argmin(diffs)]
        score = 100 * np.exp(-(delta_t**2) / (2 * 0.05**2))
        scores.append(score)
    return np.array(scores)

def _quantize_onsets_to_grid(onsets, sixteenth_grid, tolerance=0.12):
    """Maps onsets to a binary sixteenth-note grid."""
    quantized_sequence = np.zeros(len(sixteenth_grid))
    for hit in onsets:
        diffs = np.abs(sixteenth_grid - hit)
        closest_idx = np.argmin(diffs)
        if diffs[closest_idx] < tolerance:
            quantized_sequence[closest_idx] += 1
    return (quantized_sequence > 0).astype(int)

def _get_structural_segments(total_len):
    """Determines window sizes for pattern analysis based on audio length."""
    if total_len < 128:
        window_size = total_len
    elif total_len < 256:
        window_size = 128
    else:
        window_size = 256

    segments = []
    for i in range(0, total_len, window_size):
        end = min(i + window_size, total_len)
        if end - i > 16:
            segments.append((i, end))
            
    if not segments and total_len > 0:
        segments = [(0, total_len)]
    return segments

def _detect_pattern_length(segment_seq):
    """Uses autocorrelation to find the most likely musical pattern length."""
    max_lag = min(64, len(segment_seq) // 2)
    if max_lag < 4:
        return 16 if len(segment_seq) >= 16 else len(segment_seq)
        
    mean_val = np.mean(segment_seq)
    centered_seq = segment_seq - mean_val
    acf = np.correlate(centered_seq, centered_seq, mode='full')
    acf = acf[len(acf)//2:] 
    
    standard_lags = [8, 12, 16, 24, 32, 48, 64]
    valid_lags = [lag for lag in standard_lags if lag <= max_lag]
    
    if not valid_lags:
        return 16
        
    acf_normalized = [acf[lag] / (len(segment_seq) - lag) for lag in valid_lags]
    penalties = [1.0 - (lag * 0.002) for lag in valid_lags]
    adjusted_scores = [score * penalty for score, penalty in zip(acf_normalized, penalties)]
    
    return valid_lags[np.argmax(adjusted_scores)]

def _infer_intended_pattern(segment_seq, best_lag):
    """Creates a consensus pattern by folding the sequence over the pattern length."""
    num_folds = max(1.0, len(segment_seq) / best_lag)
    template = np.zeros(best_lag)
    for i in range(len(segment_seq)):
        if segment_seq[i] > 0:
            template[i % best_lag] += 1
            
    threshold = max(1, 0.55 * num_folds)
    return (template >= threshold).astype(int)

def _generate_master_grid(quantized_sequence, sixteenth_grid, segments):
    """Builds a full reference grid by unfolding inferred patterns across segments."""
    master_grid_points = []
    for start, end in segments:
        segment_seq = quantized_sequence[start:end]
        best_lag = _detect_pattern_length(segment_seq)
        intended_pattern = _infer_intended_pattern(segment_seq, best_lag)
        
        for i in range(len(segment_seq)):
            if intended_pattern[i % best_lag] == 1:
                master_grid_points.append(sixteenth_grid[start + i])
                
    return np.unique(np.array(master_grid_points))

def calculate_performance_metrics(onsets, grid_dict, duration, max_deviation=0.1):
    """Maps hits to nearest grid point using DTW and applies Gaussian scoring."""
    if len(onsets) == 0 or "sixteenth" not in grid_dict:
        return np.array([]), np.array([]), 0, 0, np.array([])

    sixteenth_grid = grid_dict["sixteenth"]
    sixteenth_grid = sixteenth_grid[sixteenth_grid <= duration]
    
    quantized_sequence = _quantize_onsets_to_grid(onsets, sixteenth_grid)
    segments = _get_structural_segments(len(quantized_sequence))
    master_grid = _generate_master_grid(quantized_sequence, sixteenth_grid, segments)
    
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
    
    scores = np.zeros(len(onsets))
    deviations = np.zeros(len(onsets))
    for gt_idx, prac_idx in final_matches:
        delta_t = onsets[prac_idx] - master_grid[gt_idx]
        score = 100 * np.exp(-(delta_t**2) / (2 * 0.05**2))
        scores[prac_idx] = score
        deviations[prac_idx] = delta_t
        
    missed_indices = [i for i in range(len(master_grid)) if i not in used_gt]
    extra_indices = [i for i in range(len(onsets)) if i not in used_prac]
        
    return scores, deviations, len(missed_indices), extra_indices, master_grid

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

def _fit_best_gmm(amplitudes):
    """Fits multiple Gaussian Mixture Models and selects the best fit using BIC."""
    X = amplitudes.reshape(-1, 1)
    best_gmm = None
    min_bic = np.inf
    for k in range(2, 5): 
        gmm = GaussianMixture(n_components=k, random_state=42).fit(X)
        bic = gmm.bic(X)
        if bic < min_bic:
            min_bic, best_gmm = bic, gmm
    return best_gmm

def _calculate_layer_scores(amplitudes, labels, means, sorted_idx):
    """Calculates consistency scores for each identified dynamic layer."""
    num_clusters = len(means)
    cluster_scores = []
    layer_info = []
    valid_weights = []

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
            valid_weights.append(means[idx])
            
            name = get_layer_name(rank, num_clusters)
            layer_info.append(f"{name}: {score:.0f}%")
            
    return cluster_scores, valid_weights, layer_info

def calculate_dynamics_metrics(amplitudes):
    """
    Advanced dynamics analysis using GMM with higher cluster resolution.
    Explores 2 to 4 dynamic layers and selects the best fit.
    """
    if len(amplitudes) < 10: # Higher cluster count needs more data points
        return 0.0, "Need more hits for detailed analysis.", np.zeros(len(amplitudes))

    best_gmm = _fit_best_gmm(amplitudes)
    labels = best_gmm.predict(amplitudes.reshape(-1, 1))
    means = best_gmm.means_.flatten()
    sorted_idx = np.argsort(means)
    
    # Map labels to their rank (0 to K-1)
    rank_map = {old_idx: new_rank for new_rank, old_idx in enumerate(sorted_idx)}
    mapped_labels = np.array([rank_map[l] for l in labels])
    
    cluster_scores, valid_weights, layer_info = _calculate_layer_scores(amplitudes, labels, means, sorted_idx)

    final_score = np.average(cluster_scores, weights=valid_weights) if cluster_scores else 0.0
    feedback = f"Found {len(means)} layers: " + " | ".join(layer_info)
    
    return round(final_score, 1), feedback, mapped_labels