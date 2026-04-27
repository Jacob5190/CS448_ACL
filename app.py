import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import audio_processor as ap

def plot_multi_row_accuracy(y, sr, onsets, grid_dict, scores, duration, row_duration=4.0):
    num_rows = int(np.ceil(duration / row_duration))
    fig, axes = plt.subplots(num_rows, 1, figsize=(14, 3 * num_rows), sharey=True)
    if num_rows == 1: axes = [axes]

    for i, ax in enumerate(axes):
        start_t, end_t = i * row_duration, min((i + 1) * row_duration, duration)
        librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.1)
        ax.set_xlim(start_t, end_t)
        
        ax.vlines(grid_dict["quarter"], -0.8, 0.8, color='black', alpha=0.4, label='1/4')
        ax.vlines(grid_dict["eighth"], -0.5, 0.5, color='gray', linestyles='--', alpha=0.2)
        ax.vlines(grid_dict["sixteenth"], -0.3, 0.3, color='lightgray', linestyles=':', alpha=0.1)
        
        mask = (onsets >= start_t) & (onsets <= end_t)
        colors = ['green' if s > 80 else 'orange' if s > 50 else 'red' for s in scores[mask]]
        ax.scatter(onsets[mask], np.ones_like(onsets[mask])*0.6, c=colors, s=70, zorder=3)
        ax.set_ylabel("Amp")
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

def plot_dynamics_analysis_gmm(onsets, amplitudes, labels):
    """
    Visualizes strike intensity with a multi-cluster color palette.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    
    cmap = plt.cm.viridis
    num_layers = len(np.unique(labels))
    
    colors = [cmap(l / (num_layers - 1 if num_layers > 1 else 1)) for l in labels]
    
    ax.bar(onsets, amplitudes, width=0.08, color=colors, alpha=0.9, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Detailed Dynamic Map ({num_layers} Layers Identified)")
    ax.set_facecolor('#f9f9f9')
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
        y, sr = ap.load_audio(uploaded_file)
        duration = len(y) / sr
        
        with st.expander("Waveform Preview"):
            fig_pre, ax_pre = plt.subplots(figsize=(10, 2))
            librosa.display.waveshow(y, sr=sr, ax=ax_pre, alpha=0.5)
            ax_pre.set_axis_off()
            st.pyplot(fig_pre)

        if st.button("Analyze Performance"):
            onsets, _, amplitudes = ap.get_onsets(y, sr)
            if len(onsets) == 0:
                st.error("No drum hits detected.")
                return

            if mode == "Auto Mode (Detects BPM Automatically)":
                final_bpm, start_offset = ap.detect_global_bpm_filtered(onsets, amplitudes, y, sr)
            else:
                final_bpm, start_offset = ap.detect_global_bpm_filtered(onsets, amplitudes, y, sr, fixed_bpm=target_bpm)

            # Scoring & Visualization Pipeline
            grid_dict = ap.generate_musical_grid(final_bpm, duration, start_offset)
            scores, deviations = ap.calculate_performance_metrics(onsets, grid_dict, duration)
            dyn_score, dyn_feedback, mapped_labels = ap.calculate_dynamics_metrics(amplitudes)

            # Dashboard
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Timing Score", f"{np.mean(scores):.1f}/100")
            c2.metric("Dynamics Score", f"{dyn_score}/100")
            c3.metric("Jitter (ms)", f"{np.std(deviations)*1000:.1f}")
            c4.metric("Analyzed BPM", f"{final_bpm}")

            st.info(f"**Feedback:** {ap.get_improvement_suggestions(deviations)}")
            st.subheader("Performance Profile")
            st.pyplot(plot_deviation_histogram(deviations))
            
            st.subheader("Dynamic Consistency (GMM Analysis)")
            fig_dyn = plot_dynamics_analysis_gmm(onsets, amplitudes, mapped_labels)
            st.pyplot(fig_dyn)
            st.caption(f"**Discovered Layers:** {dyn_feedback}")

            with st.expander("Detailed Hit-by-Hit Timeline"):
                st.pyplot(plot_multi_row_accuracy(y, sr, onsets, grid_dict, scores, duration))

if __name__ == "__main__":
    main()