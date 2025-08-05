"""
Visualization module for auto-transcription pipeline.

Contains functions for creating matplotlib visualizations with sargam pitch labels.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any


class SargamVisualizer:
    """
    Creates visualizations with sargam (Indian classical music) pitch labels.
    """
    
    def __init__(self, visualizations_dir: Path):
        """
        Initialize the visualizer.
        
        Args:
            visualizations_dir: Directory to save visualization files
        """
        self.visualizations_dir = visualizations_dir
        self.visualizations_dir.mkdir(exist_ok=True)
    
    def create_pitch_visualization(self, audio_id: int, metadata: Dict[str, Any], 
                                 pitch_results: Dict[str, Any], 
                                 confidence_thresholds: Dict[str, float]):
        """
        Create matplotlib visualization with sargam pitch labels relative to tonic.
        
        Args:
            audio_id: Unique identifier for the audio file
            metadata: Complete metadata including tonic estimation results
            pitch_results: Pitch trace extraction results
            confidence_thresholds: Algorithm-specific confidence thresholds
        """
        # Get tonic frequency from tonic estimation results
        tonic_freq = metadata["results"]["tonic_estimation"]["original"]["tonic_frequency"]
        
        # Create single axes plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        colors = ['blue', 'green', 'purple', 'orange']
        labels = ['Essentia', 'SWIPE', 'PyWorld', 'CREPE']
        keys = ['vocals_essentia', 'vocals_swipe', 'vocals_pyworld', 'vocals_crepe']
        
        for i, (key, color, label) in enumerate(zip(keys, colors, labels)):
            if key in pitch_results:
                # Load the data
                pitch_file = pitch_results[key]['pitch_data_file']
                confidence_file = pitch_results[key]['confidence_data_file']
                times_file = pitch_results[key]['times_data_file']
                
                pitch = np.load(pitch_file)
                confidence = np.load(confidence_file)
                times = np.load(times_file)
                
                # Filter for high confidence regions using algorithm-specific threshold
                confidence_threshold = confidence_thresholds[key]
                valid = (confidence > confidence_threshold) & (pitch > 0)
                
                if np.any(valid):
                    self._plot_pitch_segments(ax, times, pitch, valid, color, 0.7, label)
        
        # Set up sargam y-axis
        self._setup_sargam_yaxis(ax, tonic_freq)
        
        ax.set_title(f"Pitch Trace - {metadata['original_name']}")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Sargam (Relative to Tonic)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        pitch_file = self.visualizations_dir / f"{audio_id}_pitch_sargam.png"
        plt.savefig(pitch_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _setup_sargam_yaxis(self, ax, tonic_freq: float):
        """
        Set up y-axis with sargam labels relative to tonic frequency.
        Covers 1 octave below and 2 octaves above tonic with all 12 chromatic divisions.
        
        Args:
            ax: Matplotlib axes object
            tonic_freq: Tonic frequency in Hz
        """
        # All 12 chromatic sargam notes with their semitone offsets from Sa (tonic)
        sargam_notes = [
            ('S', 0),   # Sa (tonic) - chroma 0
            ('r', 1),   # komal Re (minor 2nd) - chroma 1
            ('R', 2),   # shuddha Re (major 2nd) - chroma 2
            ('g', 3),   # komal Ga (minor 3rd) - chroma 3
            ('G', 4),   # shuddha Ga (major 3rd) - chroma 4
            ('m', 5),   # shuddha Ma (perfect 4th) - chroma 5
            ('M', 6),   # tivra Ma (augmented 4th) - chroma 6
            ('P', 7),   # Pa (perfect 5th) - chroma 7
            ('d', 8),   # komal Dha (minor 6th) - chroma 8
            ('D', 9),   # shuddha Dha (major 6th) - chroma 9
            ('n', 10),  # komal Ni (minor 7th) - chroma 10
            ('N', 11)   # shuddha Ni (major 7th) - chroma 11
        ]
        
        # Generate frequency values and labels for 4 octaves (-1, 0, +1, +2)
        frequencies = []
        labels = []
        
        for octave in range(-1, 3):  # -1, 0, 1, 2
            for note, semitones in sargam_notes:
                # Calculate frequency: tonic * 2^(octave + semitones/12)
                freq = tonic_freq * (2 ** (octave + semitones/12))
                frequencies.append(freq)
                
                # Create label with diacritical marks for octaves
                if octave == 0:
                    # Middle octave (tonic octave) - no diacritical marks
                    label = note
                elif octave > 0:
                    # Upper octaves - dot above (using combining dot above U+0307)
                    label = note + '\u0307' * octave
                else:
                    # Lower octave - dot below (using combining dot below U+0323)
                    label = note + '\u0323'
                
                labels.append(label)
        
        # Set y-axis limits to cover the range
        min_freq = tonic_freq / 2  # 1 octave below
        max_freq = tonic_freq * 4  # 2 octaves above
        ax.set_ylim(min_freq, max_freq)
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        
        # Set all chromatic divisions as tick positions
        ax.set_yticks(frequencies)
        ax.set_yticklabels(labels, fontsize=9)
        
        # Remove default matplotlib tick labels (scientific notation)
        ax.tick_params(axis='y', which='minor', left=False, labelleft=False)
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        
        # Manually set the tick labels to only show sargam
        ax.set_yticks(frequencies)
        ax.set_yticklabels(labels, fontsize=9)
    
    def _plot_pitch_segments(self, ax, times, pitch, valid_mask, color, alpha, label=None):
        """
        Plot pitch trace as disconnected segments based on validity mask.
        This creates gaps where confidence is low instead of connecting through them.
        
        Args:
            ax: Matplotlib axes object
            times: Time axis values
            pitch: Pitch values
            valid_mask: Boolean mask for valid pitch points
            color: Plot color
            alpha: Plot transparency
            label: Legend label
        """
        # Find continuous segments of valid data
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return
        
        # Find breaks in continuity
        breaks = np.where(np.diff(valid_indices) > 1)[0]
        segment_starts = np.concatenate(([0], breaks + 1))
        segment_ends = np.concatenate((breaks + 1, [len(valid_indices)]))
        
        # Plot each continuous segment separately
        for start, end in zip(segment_starts, segment_ends):
            segment_indices = valid_indices[start:end]
            if len(segment_indices) > 1:  # Only plot segments with more than 1 point
                ax.plot(times[segment_indices], pitch[segment_indices], 
                       color, alpha=alpha, linewidth=0.8, label=label if start == 0 else None)