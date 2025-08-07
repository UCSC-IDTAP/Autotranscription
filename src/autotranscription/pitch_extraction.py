"""
Pitch extraction module for auto-transcription pipeline.

Contains implementations of various pitch estimation algorithms optimized for Indian classical music.
"""

import time
import numpy as np
import soundfile as sf
import librosa
import crepe
import pysptk
import pyworld
import essentia.standard as es
from pathlib import Path
from typing import Tuple, Dict, Any, List


class PitchExtractor:
    """
    Multi-algorithm pitch extraction for vocals.
    """
    
    def __init__(self, algorithms: List[str] = None):
        """
        Initialize the pitch extractor.
        
        Args:
            algorithms: List of algorithms to use. Default: ['essentia']
                       Available: ['essentia', 'swipe', 'pyworld', 'crepe']
        """
        self.algorithms = algorithms if algorithms is not None else ['essentia']
        
        # Algorithm-specific confidence thresholds
        self.essentia_threshold = 0.05  # Lowered for quiet sections
        self.crepe_threshold = 0.7
        self.swipe_threshold = 0.5  
        self.pyworld_threshold = 0.5
    
    def extract_all_algorithms(self, vocals_path: Path, audio_id: int, 
                             pitch_traces_dir: Path) -> Dict[str, Any]:
        """
        Extract pitch traces using all selected algorithms.
        
        Args:
            vocals_path: Path to vocals audio file
            audio_id: Unique identifier for the audio file
            pitch_traces_dir: Directory to save pitch trace data
            
        Returns:
            Dictionary with results for each algorithm
        """
        results = {}
        
        # Extract pitch trace using Essentia (if selected)
        if 'essentia' in self.algorithms:
            essentia_start = time.time()
            pitch, confidence, times = self.extract_essentia(vocals_path)
            essentia_time = time.time() - essentia_start
            
            # Save data files
            data_file = pitch_traces_dir / f"{audio_id}_vocals_essentia.npy"
            confidence_file = pitch_traces_dir / f"{audio_id}_vocals_essentia_confidence.npy"
            times_file = pitch_traces_dir / f"{audio_id}_vocals_essentia_times.npy"
            np.save(data_file, pitch)
            np.save(confidence_file, confidence)
            np.save(times_file, times)
            
            # Calculate statistics
            high_conf_frames = np.sum(confidence > self.essentia_threshold)
            conf_percentage = (high_conf_frames / len(confidence)) * 100
            
            results['vocals_essentia'] = {
                "file": str(vocals_path),
                "algorithm": "essentia_predominant_pitch_melodia",
                "pitch_data_file": str(data_file),
                "confidence_data_file": str(confidence_file),
                "times_data_file": str(times_file),
                "duration": len(times),
                "sample_rate": float(len(times) / times[-1]) if len(times) > 0 else 0.0,
                "high_confidence_frames": int(high_conf_frames),
                "confidence_percentage": conf_percentage,
                "confidence_min": float(confidence.min()),
                "confidence_max": float(confidence.max()),
                "confidence_mean": float(confidence.mean()),
                "processing_time": essentia_time
            }
        
        # Extract pitch trace using SWIPE (if selected)
        if 'swipe' in self.algorithms:
            swipe_start = time.time()
            pitch, confidence, times = self.extract_swipe(vocals_path)
            swipe_time = time.time() - swipe_start
            
            # Save data files
            data_file = pitch_traces_dir / f"{audio_id}_vocals_swipe.npy"
            confidence_file = pitch_traces_dir / f"{audio_id}_vocals_swipe_confidence.npy"
            times_file = pitch_traces_dir / f"{audio_id}_vocals_swipe_times.npy"
            np.save(data_file, pitch)
            np.save(confidence_file, confidence)
            np.save(times_file, times)
            
            # Calculate statistics
            high_conf_frames = np.sum(confidence > self.swipe_threshold)
            conf_percentage = (high_conf_frames / len(confidence)) * 100
            
            results['vocals_swipe'] = {
                "file": str(vocals_path),
                "algorithm": "swipe",
                "pitch_data_file": str(data_file),
                "confidence_data_file": str(confidence_file),
                "times_data_file": str(times_file),
                "duration": len(times),
                "sample_rate": float(len(times) / times[-1]) if len(times) > 0 else 0.0,
                "high_confidence_frames": int(high_conf_frames),
                "confidence_percentage": conf_percentage,
                "confidence_min": float(confidence.min()),
                "confidence_max": float(confidence.max()),
                "confidence_mean": float(confidence.mean()),
                "processing_time": swipe_time
            }
        
        # Extract pitch trace using PyWorld (if selected)
        if 'pyworld' in self.algorithms:
            pyworld_start = time.time()
            pitch, confidence, times = self.extract_pyworld(vocals_path)
            pyworld_time = time.time() - pyworld_start
            
            # Save data files
            data_file = pitch_traces_dir / f"{audio_id}_vocals_pyworld.npy"
            confidence_file = pitch_traces_dir / f"{audio_id}_vocals_pyworld_confidence.npy"
            times_file = pitch_traces_dir / f"{audio_id}_vocals_pyworld_times.npy"
            np.save(data_file, pitch)
            np.save(confidence_file, confidence)
            np.save(times_file, times)
            
            # Calculate statistics
            high_conf_frames = np.sum(confidence > self.pyworld_threshold)
            conf_percentage = (high_conf_frames / len(confidence)) * 100
            
            results['vocals_pyworld'] = {
                "file": str(vocals_path),
                "algorithm": "pyworld",
                "pitch_data_file": str(data_file),
                "confidence_data_file": str(confidence_file),
                "times_data_file": str(times_file),
                "duration": len(times),
                "sample_rate": float(len(times) / times[-1]) if len(times) > 0 else 0.0,
                "high_confidence_frames": int(high_conf_frames),
                "confidence_percentage": conf_percentage,
                "confidence_min": float(confidence.min()),
                "confidence_max": float(confidence.max()),
                "confidence_mean": float(confidence.mean()),
                "processing_time": pyworld_time
            }
        
        # Extract pitch trace using CREPE (if selected)
        if 'crepe' in self.algorithms:
            crepe_start = time.time()
            pitch, confidence, times = self.extract_crepe(vocals_path)
            crepe_time = time.time() - crepe_start
            
            # Save data files
            data_file = pitch_traces_dir / f"{audio_id}_vocals_crepe.npy"
            confidence_file = pitch_traces_dir / f"{audio_id}_vocals_crepe_confidence.npy"
            times_file = pitch_traces_dir / f"{audio_id}_vocals_crepe_times.npy"
            np.save(data_file, pitch)
            np.save(confidence_file, confidence)
            np.save(times_file, times)
            
            # Calculate statistics
            high_conf_frames = np.sum(confidence > self.crepe_threshold)
            conf_percentage = (high_conf_frames / len(confidence)) * 100
            
            results['vocals_crepe'] = {
                "file": str(vocals_path),
                "algorithm": "crepe",
                "pitch_data_file": str(data_file),
                "confidence_data_file": str(confidence_file),
                "times_data_file": str(times_file),
                "duration": len(times),
                "sample_rate": float(len(times) / times[-1]) if len(times) > 0 else 0.0,
                "high_confidence_frames": int(high_conf_frames),
                "confidence_percentage": conf_percentage,
                "confidence_min": float(confidence.min()),
                "confidence_max": float(confidence.max()),
                "confidence_mean": float(confidence.mean()),
                "processing_time": crepe_time
            }
        
        return results
    
    def extract_essentia(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch trace using Essentia's PredominantPitchMelodia algorithm.
        
        Parameters optimized for quiet sections:
        - voicingTolerance: 0.4 (higher than default 0.2 to catch quiet sections)
        - guessUnvoiced: True (estimate pitch in quiet/unvoiced segments)  
        - magnitudeThreshold: 20 (lower than default 40 for quiet audio sensitivity)
        
        Returns:
            Tuple of (pitch_values, confidence_values, time_axis)
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
    
    def extract_crepe(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch trace using CREPE (CNN-based pitch estimator).
        
        Returns:
            Tuple of (pitch_values, confidence_values, time_axis)
        """
        # Load audio with librosa (CREPE's preferred method)
        y, sr = librosa.load(str(audio_path), sr=16000)  # CREPE works well at 16kHz
        
        # Extract pitch using CREPE with 10ms step size
        time, frequency, confidence, activation = crepe.predict(
            y, sr, 
            step_size=10,  # 10ms step size
            verbose=0      # Suppress progress output
        )
        
        return frequency, confidence, time
    
    def extract_swipe(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch trace using SWIPE algorithm from pysptk.
        
        Returns:
            Tuple of (pitch_values, synthetic_confidence_values, time_axis)
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
    
    def extract_pyworld(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch trace using PyWorld DIO->StoneMask algorithm.
        
        Returns:
            Tuple of (pitch_values, synthetic_confidence_values, time_axis)
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
    
    def get_confidence_thresholds(self) -> Dict[str, float]:
        """
        Get algorithm-specific confidence thresholds.
        
        Returns:
            Dictionary mapping algorithm keys to their confidence thresholds
        """
        return {
            'vocals_essentia': self.essentia_threshold,
            'vocals_swipe': self.swipe_threshold,
            'vocals_pyworld': self.pyworld_threshold,
            'vocals_crepe': self.crepe_threshold
        }
