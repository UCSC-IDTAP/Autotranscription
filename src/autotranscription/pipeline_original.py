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
import crepe
import librosa
import soundfile as sf
import pysptk
import pyworld

try:
    from .audio_separator import AudioSeparator
except ImportError:
    from audio_separator import AudioSeparator


class AutoTranscriptionPipeline:
    """
    Complete pipeline for auto-transcription of Indian classical music.
    """
    
    def __init__(self, audio_separator_model: str = "htdemucs", workspace_dir: Union[str, Path] = None, 
                 algorithms: Optional[list] = None):
        """
        Initialize the pipeline.
        
        Args:
            audio_separator_model: Demucs model to use for audio separation
            workspace_dir: Directory to store all pipeline files (optional)
            algorithms: List of algorithms to run. Default: ['essentia'] 
                       Available: ['essentia', 'swipe', 'pyworld', 'crepe']
        """
        self.audio_separator = AudioSeparator(model_name=audio_separator_model)
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "pipeline_workspace"
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Default to Essentia only, but allow other algorithms as options
        self.algorithms = algorithms if algorithms is not None else ['essentia']
        
        # Algorithm-specific confidence thresholds
        self.essentia_threshold = 0.05  # Lowered from 0.1 to capture more quiet sections
        self.crepe_threshold = 0.7
        self.swipe_threshold = 0.5  # SWIPE uses 0-1 scale, no explicit confidence but threshold on pitch
        self.pyworld_threshold = 0.5  # PyWorld threshold on pitch values
        
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
        Extract pitch traces using selected algorithms (default: Essentia only).
        """
        results = {}
        
        original_path = Path(metadata["workspace_file"])
        separation_results = metadata["results"]["audio_separation"]
        vocals_path = Path(separation_results["vocals"])
        
        # Extract pitch trace from vocals track using Essentia (if selected)
        if 'essentia' in self.algorithms:
            essentia_start = time.time()
            vocals_pitch_essentia, vocals_confidence_essentia, vocals_times_essentia = self._extract_pitch_trace_essentia(vocals_path)
            essentia_vocals_time = time.time() - essentia_start
            vocals_essentia_data_file = self.pitch_traces_dir / f"{audio_id}_vocals_essentia.npy"
            vocals_essentia_confidence_file = self.pitch_traces_dir / f"{audio_id}_vocals_essentia_confidence.npy"
            vocals_essentia_times_file = self.pitch_traces_dir / f"{audio_id}_vocals_essentia_times.npy"
            np.save(vocals_essentia_data_file, vocals_pitch_essentia)
            np.save(vocals_essentia_confidence_file, vocals_confidence_essentia)
            np.save(vocals_essentia_times_file, vocals_times_essentia)
            
            # Calculate confidence statistics for Essentia
            high_conf_frames = np.sum(vocals_confidence_essentia > self.essentia_threshold)
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
                "confidence_mean": float(vocals_confidence_essentia.mean()),
                "processing_time": essentia_vocals_time
            }
        
        
        # Extract pitch trace from vocals track using SWIPE (if selected)
        if 'swipe' in self.algorithms:
            swipe_start = time.time()
            vocals_pitch_swipe, vocals_confidence_swipe, vocals_times_swipe = self._extract_pitch_trace_swipe(vocals_path)
            swipe_vocals_time = time.time() - swipe_start
            vocals_swipe_data_file = self.pitch_traces_dir / f"{audio_id}_vocals_swipe.npy"
            vocals_swipe_confidence_file = self.pitch_traces_dir / f"{audio_id}_vocals_swipe_confidence.npy"
            vocals_swipe_times_file = self.pitch_traces_dir / f"{audio_id}_vocals_swipe_times.npy"
            np.save(vocals_swipe_data_file, vocals_pitch_swipe)
            np.save(vocals_swipe_confidence_file, vocals_confidence_swipe)
            np.save(vocals_swipe_times_file, vocals_times_swipe)
            
            # Calculate confidence statistics for SWIPE
            high_conf_frames = np.sum(vocals_confidence_swipe > self.swipe_threshold)
            conf_percentage = (high_conf_frames / len(vocals_confidence_swipe)) * 100
            
            results['vocals_swipe'] = {
                "file": str(vocals_path),
                "algorithm": "swipe",
                "pitch_data_file": str(vocals_swipe_data_file),
                "confidence_data_file": str(vocals_swipe_confidence_file),
                "times_data_file": str(vocals_swipe_times_file),
                "duration": len(vocals_times_swipe),
                "sample_rate": len(vocals_times_swipe) / vocals_times_swipe[-1] if len(vocals_times_swipe) > 0 else 0,
                "high_confidence_frames": int(high_conf_frames),
                "confidence_percentage": conf_percentage,
                "confidence_min": float(vocals_confidence_swipe.min()),
                "confidence_max": float(vocals_confidence_swipe.max()),
                "confidence_mean": float(vocals_confidence_swipe.mean()),
                "processing_time": swipe_vocals_time
            }
        
        
        # Extract pitch trace from vocals track using PyWorld (if selected)
        if 'pyworld' in self.algorithms:
            pyworld_start = time.time()
            vocals_pitch_pyworld, vocals_confidence_pyworld, vocals_times_pyworld = self._extract_pitch_trace_pyworld(vocals_path)
            pyworld_vocals_time = time.time() - pyworld_start
            vocals_pyworld_data_file = self.pitch_traces_dir / f"{audio_id}_vocals_pyworld.npy"
            vocals_pyworld_confidence_file = self.pitch_traces_dir / f"{audio_id}_vocals_pyworld_confidence.npy"
            vocals_pyworld_times_file = self.pitch_traces_dir / f"{audio_id}_vocals_pyworld_times.npy"
            np.save(vocals_pyworld_data_file, vocals_pitch_pyworld)
            np.save(vocals_pyworld_confidence_file, vocals_confidence_pyworld)
            np.save(vocals_pyworld_times_file, vocals_times_pyworld)
            
            # Calculate confidence statistics for PyWorld
            high_conf_frames = np.sum(vocals_confidence_pyworld > self.pyworld_threshold)
            conf_percentage = (high_conf_frames / len(vocals_confidence_pyworld)) * 100
            
            results['vocals_pyworld'] = {
                "file": str(vocals_path),
                "algorithm": "pyworld",
                "pitch_data_file": str(vocals_pyworld_data_file),
                "confidence_data_file": str(vocals_pyworld_confidence_file),
                "times_data_file": str(vocals_pyworld_times_file),
                "duration": len(vocals_times_pyworld),
                "sample_rate": len(vocals_times_pyworld) / vocals_times_pyworld[-1] if len(vocals_times_pyworld) > 0 else 0,
                "high_confidence_frames": int(high_conf_frames),
                "confidence_percentage": conf_percentage,
                "confidence_min": float(vocals_confidence_pyworld.min()),
                "confidence_max": float(vocals_confidence_pyworld.max()),
                "confidence_mean": float(vocals_confidence_pyworld.mean()),
                "processing_time": pyworld_vocals_time
            }
        
        
        # Extract pitch trace from vocals track using CREPE (if selected)
        if 'crepe' in self.algorithms:
            crepe_start = time.time()
            vocals_pitch_crepe, vocals_confidence_crepe, vocals_times_crepe = self._extract_pitch_trace_crepe(vocals_path)
            crepe_vocals_time = time.time() - crepe_start
            vocals_crepe_data_file = self.pitch_traces_dir / f"{audio_id}_vocals_crepe.npy"
            vocals_crepe_confidence_file = self.pitch_traces_dir / f"{audio_id}_vocals_crepe_confidence.npy"
            vocals_crepe_times_file = self.pitch_traces_dir / f"{audio_id}_vocals_crepe_times.npy"
            np.save(vocals_crepe_data_file, vocals_pitch_crepe)
            np.save(vocals_crepe_confidence_file, vocals_confidence_crepe)
            np.save(vocals_crepe_times_file, vocals_times_crepe)
            
            # Calculate confidence statistics for CREPE
            high_conf_frames = np.sum(vocals_confidence_crepe > self.crepe_threshold)
            conf_percentage = (high_conf_frames / len(vocals_confidence_crepe)) * 100
            
            results['vocals_crepe'] = {
                "file": str(vocals_path),
                "algorithm": "crepe",
                "pitch_data_file": str(vocals_crepe_data_file),
                "confidence_data_file": str(vocals_crepe_confidence_file),
                "times_data_file": str(vocals_crepe_times_file),
                "duration": len(vocals_times_crepe),
                "sample_rate": len(vocals_times_crepe) / vocals_times_crepe[-1] if len(vocals_times_crepe) > 0 else 0,
                "high_confidence_frames": int(high_conf_frames),
                "confidence_percentage": conf_percentage,
                "confidence_min": float(vocals_confidence_crepe.min()),
                "confidence_max": float(vocals_confidence_crepe.max()),
                "confidence_mean": float(vocals_confidence_crepe.mean()),
                "processing_time": crepe_vocals_time
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
        
        Parameters optimized for quiet sections:
        - voicingTolerance: 0.4 (higher than default 0.2 to catch quiet sections)
        - guessUnvoiced: True (estimate pitch in quiet/unvoiced segments)  
        - magnitudeThreshold: 20 (lower than default 40 for quiet audio sensitivity)
        """
        # Load audio
        loader = es.MonoLoader(filename=str(audio_path))
        audio = loader()
        
        # Extract pitch using PredominantPitchMelodia with parameters optimized for quiet sections
        pitch_extractor = es.PredominantPitchMelodia(
            voicingTolerance=0.4,      # Higher tolerance to capture quiet sections
            guessUnvoiced=True,        # Estimate pitch in quiet/unvoiced segments
            magnitudeThreshold=20,     # Lower threshold for quiet audio sensitivity
            minFrequency=55.0,         # Appropriate for Indian classical music
            maxFrequency=1760.0        # Cover full vocal range
        )
        pitch_values, pitch_confidence = pitch_extractor(audio)
        
        # Create time axis
        hop_size = 128  # Default hop size for PredominantPitchMelodia
        sample_rate = 44100  # Assuming standard sample rate
        times = np.arange(len(pitch_values)) * hop_size / sample_rate
        
        return pitch_values, pitch_confidence, times
    
    def _extract_pitch_trace_crepe(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch trace using CREPE (CNN-based pitch estimator).
        Returns pitch values, confidence values, and time axis.
        """
        # Load audio with librosa (CREPE's preferred method)
        y, sr = librosa.load(str(audio_path), sr=16000)  # CREPE works well at 16kHz
        
        # Extract pitch using CREPE with 10ms step size (similar to PENN)
        time, frequency, confidence, activation = crepe.predict(
            y, sr, 
            step_size=10,  # 10ms step size
            verbose=0      # Suppress progress output
        )
        
        return frequency, confidence, time
    
    def _extract_pitch_trace_swipe(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch trace using SWIPE algorithm from pysptk.
        Returns pitch values, synthetic confidence values, and time axis.
        """
        # Load audio with soundfile
        wav, sr = sf.read(str(audio_path))
        
        # Ensure mono
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        
        # Extract pitch using SWIPE
        hop = int(0.01 * sr)  # 10 ms hop
        f0_swipe = pysptk.sptk.swipe(wav.astype(np.float64),
                                    fs=sr,
                                    hopsize=hop,
                                    min=55.0,     # fmin (Hz)
                                    max=1760.0,   # fmax (Hz)
                                    threshold=0.25,   # voiced/unvoiced
                                    otype="f0")
        
        # Create synthetic confidence based on pitch continuity and strength
        # SWIPE doesn't provide explicit confidence, so we estimate it
        confidence = np.ones_like(f0_swipe)
        confidence[f0_swipe <= 0] = 0.0  # Unvoiced regions get 0 confidence
        
        # Add some variability based on pitch stability
        for i in range(1, len(f0_swipe)-1):
            if f0_swipe[i] > 0:
                # Calculate local pitch stability
                prev_f0 = f0_swipe[i-1] if f0_swipe[i-1] > 0 else f0_swipe[i]
                next_f0 = f0_swipe[i+1] if f0_swipe[i+1] > 0 else f0_swipe[i]
                stability = 1.0 - min(0.5, abs(f0_swipe[i] - prev_f0) / f0_swipe[i] + abs(f0_swipe[i] - next_f0) / f0_swipe[i])
                confidence[i] = max(0.3, stability)  # Keep confidence above 0.3 for voiced regions
        
        # Create time axis
        times = np.arange(len(f0_swipe)) * hop / sr
        
        return f0_swipe, confidence, times
    
    def _extract_pitch_trace_pyworld(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch trace using PyWorld DIO->StoneMask algorithm.
        Returns pitch values, synthetic confidence values, and time axis.
        """
        # Load audio with soundfile
        wav, sr = sf.read(str(audio_path))
        
        # Ensure mono
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        
        # PyWorld expects float64
        wav_f64 = wav.astype(np.float64)
        
        # Extract F0 using DIO
        f0_dio, timeaxis = pyworld.dio(wav_f64, sr, frame_period=10.0)  # 10ms
        
        # Refine F0 using StoneMask
        f0_refined = pyworld.stonemask(wav_f64, f0_dio, timeaxis, sr)
        
        # Create synthetic confidence based on pitch values
        # PyWorld doesn't provide explicit confidence, so we estimate it
        confidence = np.ones_like(f0_refined)
        confidence[f0_refined <= 0] = 0.0  # Unvoiced regions get 0 confidence
        
        # Add variability based on pitch consistency
        for i in range(1, len(f0_refined)-1):
            if f0_refined[i] > 0:
                # Higher confidence for stable pitch regions
                if f0_refined[i-1] > 0 and f0_refined[i+1] > 0:
                    pitch_var = abs(f0_refined[i] - f0_refined[i-1]) + abs(f0_refined[i+1] - f0_refined[i])
                    confidence[i] = max(0.4, 1.0 - pitch_var / f0_refined[i])
                else:
                    confidence[i] = 0.6  # Medium confidence for isolated pitch points
        
        return f0_refined, confidence, timeaxis
    
    def _create_pitch_visualizations(self, audio_id: int, metadata: Dict[str, Any], results: Dict[str, Any]):
        """
        Create matplotlib visualization with sargam pitch labels relative to tonic.
        """
        # Get tonic frequency from tonic estimation results
        tonic_freq = metadata["results"]["tonic_estimation"]["original"]["tonic_frequency"]
        
        # Algorithm-specific confidence thresholds
        confidence_thresholds = {
            'vocals_essentia': self.essentia_threshold,
            'vocals_swipe': self.swipe_threshold,
            'vocals_pyworld': self.pyworld_threshold,
            'vocals_crepe': self.crepe_threshold
        }
        
        # Create single axes plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        colors = ['blue', 'green', 'purple', 'orange']
        labels = ['Essentia', 'SWIPE', 'PyWorld', 'CREPE']
        keys = ['vocals_essentia', 'vocals_swipe', 'vocals_pyworld', 'vocals_crepe']
        
        for i, (key, color, label) in enumerate(zip(keys, colors, labels)):
            if key in results:
                # Load the data
                pitch_file = results[key]['pitch_data_file']
                confidence_file = results[key]['confidence_data_file']
                times_file = results[key]['times_data_file']
                
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
        Covers 1 octave above and 1 octave below tonic with all 12 chromatic divisions.
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
        
        # Algorithm-specific confidence thresholds
        confidence_thresholds = {
            'vocals_essentia': self.essentia_threshold,
            'vocals_swipe': self.swipe_threshold,
            'vocals_pyworld': self.pyworld_threshold,
            'vocals_crepe': self.crepe_threshold
        }
        
        # Extract amplitude envelope from vocals track
        vocals_amplitude_envelope = self._extract_amplitude_envelope(vocals_audio, vocals_sample_rate)
        
        # Generate sine waves for vocals algorithms only
        algorithms = {
            'vocals_essentia': 'essentia_vocals', 
            'vocals_swipe': 'swipe_vocals',
            'vocals_pyworld': 'pyworld_vocals',
            'vocals_crepe': 'crepe_vocals'
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
                
                # Generate sine wave using algorithm-specific threshold
                confidence_threshold = confidence_thresholds[key]
                valid = (confidence > confidence_threshold) & (pitch > 0)
                sine_wave = self._generate_sine_overlay(pitch, valid, times, 
                                                      len(vocals_audio), vocals_sample_rate)
                
                # Apply vocals amplitude envelope to sine wave
                sine_wave_with_envelope = sine_wave * vocals_amplitude_envelope
                
                # Normalize to -6dB
                if np.max(np.abs(sine_wave_with_envelope)) > 0:
                    sine_wave_with_envelope = sine_wave_with_envelope / np.max(np.abs(sine_wave_with_envelope)) * 0.5
                
                # Save sine wave file
                sine_file = self.audio_outputs_dir / f"{audio_id}_{filename_suffix}_pitch_sine.wav"
                sine_int = (sine_wave_with_envelope * 32767).astype(np.int16)
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
    
    def _extract_amplitude_envelope(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract amplitude envelope from audio signal using the Hilbert transform.
        Returns envelope normalized to 0-1 range.
        """
        from scipy.signal import hilbert
        
        # Use Hilbert transform to get the analytic signal
        analytic_signal = hilbert(audio)
        
        # Get the amplitude envelope (magnitude of analytic signal)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Smooth the envelope to reduce noise
        # Apply a simple moving average filter
        window_size = int(0.01 * sample_rate)  # 10ms window
        if window_size > 1:
            from scipy.ndimage import uniform_filter1d
            amplitude_envelope = uniform_filter1d(amplitude_envelope.astype(np.float64), size=window_size)
        
        # Normalize to 0-1 range
        if amplitude_envelope.max() > 0:
            amplitude_envelope = amplitude_envelope / amplitude_envelope.max()
        
        return amplitude_envelope
    
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