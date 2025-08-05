"""
Auto-transcription pipeline for Indian classical music.
"""

import time
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import essentia.standard as es
import scipy.io.wavfile as wavfile
from scipy import signal
import penn
import torch

try:
    from .audio_separator import AudioSeparator
except ImportError:
    from audio_separator import AudioSeparator


class AutoTranscriptionPipeline:
    """
    Complete pipeline for auto-transcription of Indian classical music.
    """
    
    def __init__(self, audio_separator_model: str = "htdemucs", workspace_dir: Union[str, Path] = None):
        """
        Initialize the pipeline.
        
        Args:
            audio_separator_model: Demucs model to use for audio separation
            workspace_dir: Directory to store all pipeline files (optional)
        """
        self.audio_separator = AudioSeparator(model_name=audio_separator_model)
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "pipeline_workspace"
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.audio_dir = self.workspace_dir / "audio"
        self.metadata_dir = self.workspace_dir / "metadata"
        self.separated_dir = self.workspace_dir / "separated"
        self.data_dir = self.workspace_dir / "data"
        self.pitch_traces_dir = self.data_dir / "pitch_traces"
        self.visualizations_dir = self.workspace_dir / "visualizations"
        self.audio_outputs_dir = self.workspace_dir / "audio_outputs"
        
        for dir_path in [self.audio_dir, self.metadata_dir, self.separated_dir, 
                        self.data_dir, self.pitch_traces_dir, self.visualizations_dir,
                        self.audio_outputs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def get_audio_id_for_file(self, input_path: Path) -> int:
        """Get existing ID for a file or create a new one."""
        input_path_str = str(input_path.resolve())
        
        # Check if this file already exists in metadata
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                audio_id = int(metadata_file.stem)
                metadata = self.load_metadata(audio_id)
                if metadata and metadata.get("original_file") == input_path_str:
                    return audio_id
            except ValueError:
                continue
        
        # File not found, get next available ID
        existing_ids = []
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                existing_ids.append(int(metadata_file.stem))
            except ValueError:
                continue
        return max(existing_ids, default=-1) + 1
    
    def load_metadata(self, audio_id: int) -> Optional[Dict[str, Any]]:
        """Load metadata for a given audio ID."""
        metadata_file = self.metadata_dir / f"{audio_id}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_metadata(self, audio_id: int, metadata: Dict[str, Any]):
        """Save metadata for a given audio ID."""
        metadata_file = self.metadata_dir / f"{audio_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run(self, input_file: Union[str, Path], audio_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete auto-transcription pipeline.
        
        Args:
            input_file: Path to the input audio file
            audio_id: Optional numeric ID to use (if None, will generate new ID)
            
        Returns:
            dict: Complete results from all pipeline stages
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Get or assign audio ID
        if audio_id is None:
            audio_id = self.get_audio_id_for_file(input_path)
        
        # Load existing metadata or create new
        metadata = self.load_metadata(audio_id)
        if metadata is None:
            # Create new metadata entry
            workspace_audio_path = self.audio_dir / f"{audio_id}{input_path.suffix}"
            if not workspace_audio_path.exists():
                shutil.copy2(input_path, workspace_audio_path)
            
            metadata = {
                "id": audio_id,
                "original_file": str(input_path),
                "original_name": input_path.name,
                "workspace_file": str(workspace_audio_path),
                "file_extension": input_path.suffix,
                "stages_completed": [],
                "results": {}
            }
        
        pipeline_start = time.time()
        
        # Step 1: Audio Separation
        if "audio_separation" not in metadata["stages_completed"]:
            separation_results = self._run_audio_separation(audio_id, metadata)
            metadata["results"]["audio_separation"] = separation_results
            metadata["stages_completed"].append("audio_separation")
            self.save_metadata(audio_id, metadata)
        else:
            separation_results = metadata["results"]["audio_separation"]
        
        # Step 2: Tonic Estimation
        if "tonic_estimation" not in metadata["stages_completed"]:
            tonic_results = self._run_tonic_estimation(audio_id, metadata)
            metadata["results"]["tonic_estimation"] = tonic_results
            metadata["stages_completed"].append("tonic_estimation")
            self.save_metadata(audio_id, metadata)
        else:
            tonic_results = metadata["results"]["tonic_estimation"]
        
        # Step 3: Pitch Trace Extraction
        if "pitch_trace" not in metadata["stages_completed"]:
            pitch_results = self._run_pitch_trace_extraction(audio_id, metadata)
            metadata["results"]["pitch_trace"] = pitch_results
            metadata["stages_completed"].append("pitch_trace")
            self.save_metadata(audio_id, metadata)
        else:
            pitch_results = metadata["results"]["pitch_trace"]
        
        pipeline_time = time.time() - pipeline_start
        metadata["last_run_time"] = pipeline_time
        self.save_metadata(audio_id, metadata)
        
        return metadata
    
    def _run_audio_separation(self, audio_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run audio separation step.
        """
        workspace_audio_path = Path(metadata["workspace_file"])
        separation_output_dir = self.separated_dir / str(audio_id)
        
        results = self.audio_separator.separate_vocals(workspace_audio_path, separation_output_dir)
        
        # Update paths to be relative to workspace
        results["vocals"] = str(results["vocals"])
        results["instrumental"] = str(results["instrumental"])
        results["output_dir"] = str(results["output_dir"])
        
        return results
    
    def _run_tonic_estimation(self, _audio_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run tonic estimation using Essentia's TonicIndianArtMusic algorithm.
        """
        results = {}
        
        original_path = Path(metadata["workspace_file"])
        separation_results = metadata["results"]["audio_separation"]
        instrumental_path = Path(separation_results["instrumental"])
        
        # Test on original track
        print("Estimating tonic on original track...")
        original_tonic = self._estimate_tonic(original_path)
        results['original'] = {
            "file": str(original_path),
            "tonic_frequency": original_tonic,
            "tonic_note": self._frequency_to_note(original_tonic)
        }
        
        # Test on separated instrumental track
        instrumental_tonic = self._estimate_tonic(instrumental_path)
        results['instrumental'] = {
            "file": str(instrumental_path),
            "tonic_frequency": instrumental_tonic,
            "tonic_note": self._frequency_to_note(instrumental_tonic)
        }
        
        # Compare results
        frequency_diff = abs(original_tonic - instrumental_tonic)
        cents_diff = 1200 * abs(original_tonic - instrumental_tonic) / min(original_tonic, instrumental_tonic) if min(original_tonic, instrumental_tonic) > 0 else 0
        
        results['comparison'] = {
            "frequency_difference": frequency_diff,
            "cents_difference": cents_diff
        }
        
        return results
    
    def _run_pitch_trace_extraction(self, audio_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract pitch traces using Essentia's PredominantPitchMelodia algorithm.
        """
        results = {}
        
        original_path = Path(metadata["workspace_file"])
        separation_results = metadata["results"]["audio_separation"]
        vocals_path = Path(separation_results["vocals"])
        
        # Extract pitch trace from original track using Essentia
        original_pitch_essentia, original_confidence_essentia, original_times_essentia = self._extract_pitch_trace_essentia(original_path)
        original_essentia_data_file = self.pitch_traces_dir / f"{audio_id}_original_essentia.npy"
        original_essentia_confidence_file = self.pitch_traces_dir / f"{audio_id}_original_essentia_confidence.npy"
        original_essentia_times_file = self.pitch_traces_dir / f"{audio_id}_original_essentia_times.npy"
        np.save(original_essentia_data_file, original_pitch_essentia)
        np.save(original_essentia_confidence_file, original_confidence_essentia)
        np.save(original_essentia_times_file, original_times_essentia)
        
        # Calculate confidence statistics for Essentia
        high_conf_frames = np.sum(original_confidence_essentia > 0.1)
        conf_percentage = (high_conf_frames / len(original_confidence_essentia)) * 100
        
        results['original_essentia'] = {
            "file": str(original_path),
            "algorithm": "essentia_predominant_pitch_melodia",
            "pitch_data_file": str(original_essentia_data_file),
            "confidence_data_file": str(original_essentia_confidence_file),
            "times_data_file": str(original_essentia_times_file),
            "duration": len(original_times_essentia),
            "sample_rate": len(original_times_essentia) / original_times_essentia[-1] if len(original_times_essentia) > 0 else 0,
            "high_confidence_frames": int(high_conf_frames),
            "confidence_percentage": conf_percentage,
            "confidence_min": float(original_confidence_essentia.min()),
            "confidence_max": float(original_confidence_essentia.max()),
            "confidence_mean": float(original_confidence_essentia.mean())
        }
        
        # Extract pitch trace from vocals track using Essentia
        vocals_pitch_essentia, vocals_confidence_essentia, vocals_times_essentia = self._extract_pitch_trace_essentia(vocals_path)
        vocals_essentia_data_file = self.pitch_traces_dir / f"{audio_id}_vocals_essentia.npy"
        vocals_essentia_confidence_file = self.pitch_traces_dir / f"{audio_id}_vocals_essentia_confidence.npy"
        vocals_essentia_times_file = self.pitch_traces_dir / f"{audio_id}_vocals_essentia_times.npy"
        np.save(vocals_essentia_data_file, vocals_pitch_essentia)
        np.save(vocals_essentia_confidence_file, vocals_confidence_essentia)
        np.save(vocals_essentia_times_file, vocals_times_essentia)
        
        # Calculate confidence statistics for Essentia
        high_conf_frames = np.sum(vocals_confidence_essentia > 0.1)
        conf_percentage = (high_conf_frames / len(vocals_confidence_essentia)) * 100
        
        results['vocals_essentia'] = {
            "file": str(vocals_path),
            "algorithm": "essentia_predominant_pitch_melodia",
            "pitch_data_file": str(vocals_essentia_data_file),
            "confidence_data_file": str(vocals_essentia_confidence_file),
            "times_data_file": str(vocals_essentia_times_file),
            "duration": len(vocals_times_essentia),
            "sample_rate": len(vocals_times_essentia) / vocals_times_essentia[-1] if len(vocals_times_essentia) > 0 else 0,
            "high_confidence_frames": int(high_conf_frames),
            "confidence_percentage": conf_percentage,
            "confidence_min": float(vocals_confidence_essentia.min()),
            "confidence_max": float(vocals_confidence_essentia.max()),
            "confidence_mean": float(vocals_confidence_essentia.mean())
        }
        
        # Extract pitch trace from original track using PENN
        original_pitch_penn, original_confidence_penn, original_times_penn = self._extract_pitch_trace_penn(original_path)
        original_penn_data_file = self.pitch_traces_dir / f"{audio_id}_original_penn.npy"
        original_penn_confidence_file = self.pitch_traces_dir / f"{audio_id}_original_penn_confidence.npy"
        original_penn_times_file = self.pitch_traces_dir / f"{audio_id}_original_penn_times.npy"
        np.save(original_penn_data_file, original_pitch_penn)
        np.save(original_penn_confidence_file, original_confidence_penn)
        np.save(original_penn_times_file, original_times_penn)
        
        # Calculate confidence statistics for PENN
        high_conf_frames = np.sum(original_confidence_penn > 0.1)
        conf_percentage = (high_conf_frames / len(original_confidence_penn)) * 100
        
        results['original_penn'] = {
            "file": str(original_path),
            "algorithm": "penn",
            "pitch_data_file": str(original_penn_data_file),
            "confidence_data_file": str(original_penn_confidence_file),
            "times_data_file": str(original_penn_times_file),
            "duration": len(original_times_penn),
            "sample_rate": len(original_times_penn) / original_times_penn[-1] if len(original_times_penn) > 0 else 0,
            "high_confidence_frames": int(high_conf_frames),
            "confidence_percentage": conf_percentage,
            "confidence_min": float(original_confidence_penn.min()),
            "confidence_max": float(original_confidence_penn.max()),
            "confidence_mean": float(original_confidence_penn.mean())
        }
        
        # Extract pitch trace from vocals track using PENN
        vocals_pitch_penn, vocals_confidence_penn, vocals_times_penn = self._extract_pitch_trace_penn(vocals_path)
        vocals_penn_data_file = self.pitch_traces_dir / f"{audio_id}_vocals_penn.npy"
        vocals_penn_confidence_file = self.pitch_traces_dir / f"{audio_id}_vocals_penn_confidence.npy"
        vocals_penn_times_file = self.pitch_traces_dir / f"{audio_id}_vocals_penn_times.npy"
        np.save(vocals_penn_data_file, vocals_pitch_penn)
        np.save(vocals_penn_confidence_file, vocals_confidence_penn)
        np.save(vocals_penn_times_file, vocals_times_penn)
        
        # Calculate confidence statistics for PENN
        high_conf_frames = np.sum(vocals_confidence_penn > 0.1)
        conf_percentage = (high_conf_frames / len(vocals_confidence_penn)) * 100
        
        results['vocals_penn'] = {
            "file": str(vocals_path),
            "algorithm": "penn",
            "pitch_data_file": str(vocals_penn_data_file),
            "confidence_data_file": str(vocals_penn_confidence_file),
            "times_data_file": str(vocals_penn_times_file),
            "duration": len(vocals_times_penn),
            "sample_rate": len(vocals_times_penn) / vocals_times_penn[-1] if len(vocals_times_penn) > 0 else 0,
            "high_confidence_frames": int(high_conf_frames),
            "confidence_percentage": conf_percentage,
            "confidence_min": float(vocals_confidence_penn.min()),
            "confidence_max": float(vocals_confidence_penn.max()),
            "confidence_mean": float(vocals_confidence_penn.mean())
        }
        
        # Create visualizations
        self._create_pitch_visualizations(audio_id, metadata, results)
        
        # Create audio outputs with sine wave overlays
        self._create_pitch_audio_outputs(audio_id, vocals_path, results)
        
        return results
    
    def _extract_pitch_trace_essentia(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch trace using Essentia's PredominantPitchMelodia algorithm.
        Returns pitch values, confidence values, and time axis.
        """
        # Load audio
        loader = es.MonoLoader(filename=str(audio_path))
        audio = loader()
        
        # Extract pitch using PredominantPitchMelodia
        pitch_extractor = es.PredominantPitchMelodia()
        pitch_values, pitch_confidence = pitch_extractor(audio)
        
        # Create time axis
        hop_size = 128  # Default hop size for PredominantPitchMelodia
        sample_rate = 44100  # Assuming standard sample rate
        times = np.arange(len(pitch_values)) * hop_size / sample_rate
        
        return pitch_values, pitch_confidence, times
    
    def _extract_pitch_trace_penn(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch trace using PENN (Pitch Estimation using Neural Networks).
        Returns pitch values, confidence values, and time axis.
        """
        import subprocess
        import tempfile
        
        # Always convert to ensure proper format for PENN (mono, 22050 Hz)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = Path(temp_wav.name)
        
        # Convert to WAV using ffmpeg
        subprocess.run([
            'ffmpeg', '-i', str(audio_path), 
            '-ar', '22050',  # Resample to PENN's expected rate
            '-ac', '1',      # Convert to mono
            '-acodec', 'pcm_s16le',  # Ensure proper WAV encoding
            '-y',            # Overwrite output
            str(temp_wav_path)
        ], check=True, capture_output=True)
        
        try:
            # Extract pitch using PENN from file path
            pitch_values, confidence_values = penn.from_file(str(temp_wav_path))
            
            # Convert results to numpy if they're tensors and flatten if needed
            if hasattr(pitch_values, 'detach'):
                pitch_values = pitch_values.detach().cpu().numpy().flatten()
            if hasattr(confidence_values, 'detach'):
                confidence_values = confidence_values.detach().cpu().numpy().flatten()
            
            # Create time axis - PENN uses 10ms hop size by default
            hop_size_seconds = 0.01  # 10ms
            times = np.arange(len(pitch_values)) * hop_size_seconds
            
            return pitch_values, confidence_values, times
            
        finally:
            # Clean up temporary file
            if temp_wav_path.exists():
                temp_wav_path.unlink()
    
    def _create_pitch_visualizations(self, audio_id: int, metadata: Dict[str, Any], results: Dict[str, Any]):
        """
        Create matplotlib visualizations comparing all 4 pitch traces with logarithmic frequency scale.
        """
        confidence_threshold = 0.1
        
        # Create comparison plot with all 4 algorithms
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        colors = ['blue', 'red', 'green', 'orange']
        labels = ['Essentia Original', 'Essentia Vocals', 'PENN Original', 'PENN Vocals']
        keys = ['original_essentia', 'vocals_essentia', 'original_penn', 'vocals_penn']
        
        for i, (key, color, label) in enumerate(zip(keys, colors, labels)):
            if key in results:
                # Load the data
                pitch_file = results[key]['pitch_data_file']
                confidence_file = results[key]['confidence_data_file']
                times_file = results[key]['times_data_file']
                
                pitch = np.load(pitch_file)
                confidence = np.load(confidence_file)
                times = np.load(times_file)
                
                # Filter for high confidence regions
                valid = (confidence > confidence_threshold) & (pitch > 0)
                
                if np.any(valid):
                    self._plot_pitch_segments(ax, times, pitch, valid, color, 0.7, label)
        
        ax.set_title(f"Pitch Trace Comparison - {metadata['original_name']} (All Algorithms)")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(80, 2000)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        comparison_file = self.visualizations_dir / f"{audio_id}_pitch_comparison_all.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create individual algorithm comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        axes = [ax1, ax2, ax3, ax4]
        titles = ['Essentia - Original Track', 'Essentia - Vocals Track', 'PENN - Original Track', 'PENN - Vocals Track']
        
        for i, (ax, key, title, color) in enumerate(zip(axes, keys, titles, colors)):
            if key in results:
                # Load the data
                pitch_file = results[key]['pitch_data_file']
                confidence_file = results[key]['confidence_data_file']
                times_file = results[key]['times_data_file']
                
                pitch = np.load(pitch_file)
                confidence = np.load(confidence_file)
                times = np.load(times_file)
                
                # Filter for high confidence regions
                valid = (confidence > confidence_threshold) & (pitch > 0)
                
                if np.any(valid):
                    self._plot_pitch_segments(ax, times, pitch, valid, color, 0.7)
                
                ax.set_title(f"{title} (Confidence > {confidence_threshold})")
                ax.set_ylabel('Frequency (Hz)')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(80, 2000)
                ax.set_yscale('log')
                
                if i >= 2:  # Bottom row
                    ax.set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        individual_file = self.visualizations_dir / f"{audio_id}_pitch_individual.png"
        plt.savefig(individual_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pitch_segments(self, ax, times, pitch, valid_mask, color, alpha, label=None):
        """
        Plot pitch trace as disconnected segments based on validity mask.
        This creates gaps where confidence is low instead of connecting through them.
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
    
    def _create_pitch_audio_outputs(self, audio_id: int, vocals_path: Path, results: Dict[str, Any]):
        """
        Create sine wave audio files for all 4 pitch traces.
        """
        # Load vocals audio for reference length and sample rate
        vocals_sample_rate, vocals_audio = wavfile.read(vocals_path)
        
        # Ensure mono
        if len(vocals_audio.shape) > 1:
            vocals_audio = vocals_audio.mean(axis=1)
        
        confidence_threshold = 0.1
        
        # Generate sine waves for all algorithms
        algorithms = {
            'original_essentia': 'essentia_original',
            'vocals_essentia': 'essentia_vocals', 
            'original_penn': 'penn_original',
            'vocals_penn': 'penn_vocals'
        }
        
        for key, filename_suffix in algorithms.items():
            if key in results:
                # Load pitch trace data
                pitch_file = results[key]['pitch_data_file']
                confidence_file = results[key]['confidence_data_file']
                times_file = results[key]['times_data_file']
                
                pitch = np.load(pitch_file)
                confidence = np.load(confidence_file)
                times = np.load(times_file)
                
                # Generate sine wave
                valid = (confidence > confidence_threshold) & (pitch > 0)
                sine_wave = self._generate_sine_overlay(pitch, valid, times, 
                                                      len(vocals_audio), vocals_sample_rate)
                
                # Normalize to -6dB
                if np.max(np.abs(sine_wave)) > 0:
                    sine_wave = sine_wave / np.max(np.abs(sine_wave)) * 0.5
                
                # Save sine wave file
                sine_file = self.audio_outputs_dir / f"{audio_id}_{filename_suffix}_pitch_sine.wav"
                sine_int = (sine_wave * 32767).astype(np.int16)
                wavfile.write(sine_file, vocals_sample_rate, sine_int)
    
    def _generate_sine_overlay(self, pitch_values: np.ndarray, valid_mask: np.ndarray, 
                             times: np.ndarray, audio_length: int, sample_rate: int) -> np.ndarray:
        """
        Generate a sine wave overlay based on pitch trace data with continuous frequency modulation.
        """
        sine_wave = np.zeros(audio_length)
        
        # Create a continuous time axis for the audio
        audio_times = np.arange(audio_length) / sample_rate
        
        # Interpolate pitch values to match audio sample rate
        # Only use valid pitch values for interpolation
        valid_indices = np.where(valid_mask & (pitch_values > 0))[0]
        
        if len(valid_indices) == 0:
            return sine_wave  # Return silence if no valid pitch
        
        valid_times = times[valid_indices]
        valid_pitches = pitch_values[valid_indices]
        
        # Interpolate to get pitch for every audio sample
        # Use nearest-neighbor interpolation to avoid smoothing between distant pitch segments
        interpolated_pitch = np.interp(audio_times, valid_times, valid_pitches)
        
        # Create mask for where we have valid pitch data (within range of valid segments)
        audio_valid_mask = np.zeros(audio_length, dtype=bool)
        
        # Mark samples as valid if they're close to any valid pitch frame
        hop_size = 128
        tolerance = hop_size / sample_rate * 2  # Allow some tolerance around each frame
        
        for valid_idx in valid_indices:
            frame_time = times[valid_idx]
            start_time = frame_time - tolerance
            end_time = frame_time + tolerance
            
            start_sample = max(0, int(start_time * sample_rate))
            end_sample = min(audio_length, int(end_time * sample_rate))
            
            audio_valid_mask[start_sample:end_sample] = True
        
        # Generate continuous sine wave with phase continuity
        phase = 0.0
        dt = 1.0 / sample_rate
        
        for i in range(audio_length):
            if audio_valid_mask[i]:
                frequency = interpolated_pitch[i]
                sine_wave[i] = np.sin(phase)
                phase += 2 * np.pi * frequency * dt
                # Keep phase in reasonable range to avoid numerical issues
                if phase > 2 * np.pi:
                    phase -= 2 * np.pi
            else:
                # Reset phase during silent periods to avoid discontinuities
                phase = 0.0
        
        return sine_wave
    
    def _estimate_tonic(self, audio_path: Path) -> float:
        """
        Estimate tonic frequency using Essentia's TonicIndianArtMusic algorithm.
        """
        # Load audio
        loader = es.MonoLoader(filename=str(audio_path))
        audio = loader()
        
        # Estimate tonic
        tonic_estimator = es.TonicIndianArtMusic()
        tonic_frequency = tonic_estimator(audio)
        
        return tonic_frequency
    
    def _frequency_to_note(self, frequency: float) -> str:
        """
        Convert frequency to approximate note name.
        """
        import math
        
        # A4 = 440 Hz
        A4 = 440.0
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Calculate semitones from A4
        semitones_from_A4 = round(12 * math.log2(frequency / A4))
        
        # Simple approximation - this could be improved for Indian music
        note_index = (9 + semitones_from_A4) % 12  # A is at index 9
        octave = 4 + (9 + semitones_from_A4) // 12
        
        return f"{note_names[note_index]}{octave}"
    
    def list_audio_files(self) -> Dict[int, Dict[str, Any]]:
        """
        List all audio files in the workspace with their metadata.
        """
        audio_files = {}
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                audio_id = int(metadata_file.stem)
                metadata = self.load_metadata(audio_id)
                if metadata:
                    audio_files[audio_id] = {
                        "original_name": metadata["original_name"],
                        "stages_completed": metadata["stages_completed"]
                    }
            except ValueError:
                continue
        return audio_files


def run_pipeline(input_file: Union[str, Path], 
                workspace_dir: Union[str, Path] = None,
                audio_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function to run the complete pipeline.
    
    Args:
        input_file: Path to the input audio file
        workspace_dir: Directory to store all pipeline files (optional)
        audio_id: Optional numeric ID to use (if None, will generate new ID)
        
    Returns:
        dict: Complete metadata with all pipeline results
    """
    pipeline = AutoTranscriptionPipeline(workspace_dir=workspace_dir)
    return pipeline.run(input_file, audio_id)


if __name__ == "__main__":
    # Test pipeline
    project_root = Path(__file__).parent.parent.parent
    audio_file = project_root / "audio" / "Kishori Amonkar - Babul Mora_1min.mp3"
    workspace_dir = project_root / "pipeline_workspace"
    
    if audio_file.exists():
        results = run_pipeline(audio_file, workspace_dir)