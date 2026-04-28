import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import base64
import os

import audio_processor as ap


def plot_multi_row_accuracy(y, sr, onsets, grid_dict, scores, labels, duration, master_grid, row_duration=4.0):
    num_rows = int(np.ceil(duration / row_duration))
    fig, axes = plt.subplots(num_rows, 1, figsize=(14, 3 * num_rows), sharey=True)
    if num_rows == 1: axes = [axes]

    num_layers = len(np.unique(labels)) if labels is not None else 1
    
    for i, ax in enumerate(axes):
        start_t, end_t = i * row_duration, min((i + 1) * row_duration, duration)
        librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.1)
        ax.set_xlim(start_t, end_t)
        
        ax.vlines(grid_dict["quarter"], -0.8, 0.8, color='black', alpha=0.15, label='Beat Grid')
        ax.vlines(master_grid, -1.0, 1.0, color='blue', alpha=0.2, linestyles='-', lw=2, label='Intended (Master Grid)')
        
        mask = (onsets >= start_t) & (onsets <= end_t)
        
        # Color logic: Green (>80), Orange (>50), Red (Error), Blue (Ghost Note)
        colors = []
        for s, l in zip(scores[mask], labels[mask] if labels is not None else [1]*np.sum(mask)):
            if num_layers > 1 and l == 0:
                colors.append('blue')
            elif s > 80:
                colors.append('green')
            elif s > 50:
                colors.append('orange')
            else:
                colors.append('red')
                
        ax.scatter(onsets[mask], np.ones_like(onsets[mask])*0.6, c=colors, s=70, zorder=3)
        ax.set_ylabel("Amp")
        if i == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', alpha=0.3, lw=2, label='Discovered Pattern (Master Grid)'),
                Line2D([0], [0], marker='o', color='w', label='Great Timing (>80)', markerfacecolor='green', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Acceptable (>50)', markerfacecolor='orange', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='Timing Error / Flub', markerfacecolor='red', markersize=8),
            ]
            if num_layers > 1:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Ghost Note (Stylistic)', markerfacecolor='blue', markersize=8))
            ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
            
        if i != num_rows - 1: ax.set_xticks([])
    
    plt.tight_layout()
    return fig

def plot_deviation_histogram(deviations):
    fig, ax = plt.subplots(figsize=(8, 3))
    n, bins, patches = ax.hist(deviations * 1000, bins=30, range=(-100, 100), alpha=0.7, edgecolor='white')
    for b, p in zip(bins, patches):
        p.set_facecolor('skyblue' if b < 0 else 'salmon')
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_title("Timing Tendency Profile (ms)")
    ax.set_xlabel("Rushing (Early) | Dragging (Late)")
    return fig

def plot_dynamics_analysis(onsets, amplitudes):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(onsets, amplitudes, width=0.08, color=plt.cm.magma(amplitudes), alpha=0.9, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_title("Velocity Map (Strike Intensity)")
    ax.set_facecolor('#f9f9f9')
    return fig

def plot_decay_analysis(onsets, decay_rates):
    fig, ax = plt.subplots(figsize=(10, 3))
    # Normalize for better visualization if needed
    max_decay = np.max(decay_rates) if len(decay_rates) > 0 and np.max(decay_rates) > 0 else 1.0
    ax.bar(onsets, decay_rates / max_decay, width=0.08, color='purple', alpha=0.7, edgecolor='black')
    ax.set_title("Relative Decay Profile (Sustain Consistency)")
    ax.set_ylabel("Relative Decay")
    ax.set_facecolor('#f9f9f9')
    return fig

def plot_dynamics_analysis_gmm(onsets, amplitudes, labels):
    """
    Visualizes strike intensity with a multi-cluster color palette.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    
    cmap = plt.cm.viridis
    unique_labels = np.unique(labels)
    num_layers = len(unique_labels)
    
    colors = [cmap(l / (num_layers - 1 if num_layers > 1 else 1)) for l in labels]
    
    bars = ax.bar(onsets, amplitudes, width=0.08, color=colors, alpha=0.9, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Detailed Dynamic Map ({num_layers} Layers Identified)")
    ax.set_facecolor('#f9f9f9')
    
    # Legend for layers
    from matplotlib.patches import Patch
    def get_layer_name(rank, total):
        if rank == 0: return "Ghost Layer"
        if rank == total - 1: return "Accent Layer"
        return f"Layer {rank+1}"
        
    legend_elements = [
        Patch(facecolor=cmap(l / (num_layers - 1 if num_layers > 1 else 1)), label=get_layer_name(l, num_layers))
        for l in unique_labels
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
    
    return fig


def main():
    st.set_page_config(page_title="Percussion Practice Analyzer", layout="wide")
    st.title("Percussion Practice Tool")
    st.caption("CS448 Project")

    st.sidebar.header("Session Setup")
    uploaded_file = st.sidebar.file_uploader("Upload Practice Recording", type=["mp3", "wav"])
    mode = st.sidebar.radio("Metronome Mode", ["Auto Mode (Detects BPM Automatically)", "Manual Mode (Input BPM)"])
    
    target_bpm = 120
    if mode == "Manual Mode (Input BPM)":
        target_bpm = st.sidebar.number_input("Input Target BPM", 40, 250, 120)

    if uploaded_file:
        st.audio(uploaded_file)
        y, sr = ap.load_audio(uploaded_file)
        duration = len(y) / sr
        
        with st.expander("Waveform Preview"):
            fig_pre, ax_pre = plt.subplots(figsize=(10, 2))
            librosa.display.waveshow(y, sr=sr, ax=ax_pre, alpha=0.5)
            ax_pre.set_axis_off()
            st.pyplot(fig_pre)

        if st.button("Analyze Performance"):
            onsets, _, amplitudes, decay_rates = ap.get_onsets(y, sr)
            if len(onsets) == 0:
                st.error("No drum hits detected.")
                return

            if mode == "Auto Mode (Detects BPM Automatically)":
                final_bpm, start_offset = ap.detect_global_bpm_filtered(onsets, amplitudes, y, sr)
            else:
                final_bpm, start_offset = ap.detect_global_bpm_filtered(onsets, amplitudes, y, sr, fixed_bpm=target_bpm)

            # Scoring & Visualization Pipeline
            grid_dict = ap.generate_musical_grid(final_bpm, duration, start_offset)
            scores, deviations, num_missed, extra_indices, master_grid = ap.calculate_performance_metrics(onsets, grid_dict, duration)
            dyn_score, dyn_feedback, mapped_labels = ap.calculate_dynamics_metrics(amplitudes)

            num_ghosts = 0
            num_flubs = 0
            if len(np.unique(mapped_labels)) > 1:
                for idx in extra_indices:
                    if mapped_labels[idx] == 0:
                        num_ghosts += 1
                    else:
                        num_flubs += 1
            else:
                num_flubs = len(extra_indices)

            # Dashboard
            st.divider()
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            valid_scores = scores[scores > 0]
            c1.metric("Timing Score", f"{np.mean(valid_scores):.1f}/100" if len(valid_scores) > 0 else "0.0/100")
            c2.metric("Dynamics Score", f"{dyn_score}/100")
            c3.metric("Analyzed BPM", f"{final_bpm}")
            c4.metric("Missed Hits", f"{num_missed}")
            c5.metric("Ghost Notes", f"{num_ghosts}")
            c6.metric("Flubs", f"{num_flubs}")

            st.info(f"**Feedback:** {ap.get_improvement_suggestions(deviations, num_missed, num_flubs)}")
            st.subheader("Performance Profile")
            st.pyplot(plot_deviation_histogram(deviations))
            
            st.subheader("Dynamic Consistency (GMM Analysis)")
            fig_dyn = plot_dynamics_analysis_gmm(onsets, amplitudes, mapped_labels)
            st.pyplot(fig_dyn)
            st.caption(f"**Discovered Layers:** {dyn_feedback}")

            st.subheader("Decay Rate Profile")
            st.pyplot(plot_decay_analysis(onsets, decay_rates))
            st.caption("Higher values indicate longer sustain; inconsistent sustain can suggest technique issues.")

            with st.expander("Detailed Hit-by-Hit Timeline"):
                st.pyplot(plot_multi_row_accuracy(y, sr, onsets, grid_dict, scores, mapped_labels, duration, master_grid))

            # Advanced Debugging: Interactive Player with Master Grid Overlay
            st.subheader("Master Grid  (Interactive)")

            
            audio_base64 = base64.b64encode(uploaded_file.getvalue()).decode()
            markers_js = ",".join([f"{{time: {t}, label: 'BEAT'}}" for t in master_grid])
            
            template_path = os.path.join(os.path.dirname(__file__), "master_grid.html")
            with open(template_path, "r") as f:
                html_template = f.read()
            
            wavesurfer_html = html_template.replace("{{AUDIO_BASE64}}", audio_base64).replace("{{MARKERS_JS}}", markers_js)
            
            st.components.v1.html(wavesurfer_html, height=350)
            st.caption(f"The blue vertical lines represent the **Master Grid** ({len(master_grid)} points detected). Use the zoom slider to see hits with millisecond precision.")


if __name__ == "__main__":
    main()