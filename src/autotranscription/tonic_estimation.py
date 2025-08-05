"""
Tonic estimation module for auto-transcription pipeline.

Contains functions for estimating tonic frequency in Indian classical music.
"""

import math
import essentia.standard as es
from pathlib import Path
from typing import Dict, Any


class TonicEstimator:
    """
    Estimates tonic frequency using Essentia's TonicIndianArtMusic algorithm.
    """
    
    def estimate_tonic_from_files(self, original_path: Path, instrumental_path: Path) -> Dict[str, Any]:
        """
        Estimate tonic frequency from both original and instrumental tracks.
        
        Args:
            original_path: Path to original audio file
            instrumental_path: Path to separated instrumental track
            
        Returns:
            Dictionary with tonic estimation results and comparison
        """
        results = {}
        
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
    
    def _estimate_tonic(self, audio_path: Path) -> float:
        """
        Estimate tonic frequency using Essentia's TonicIndianArtMusic algorithm.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Estimated tonic frequency in Hz
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
        Convert frequency to approximate Western note name.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Note name with octave (e.g., "C4", "F#3")
        """
        # A4 = 440 Hz
        A4 = 440.0
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Calculate semitones from A4
        semitones_from_A4 = round(12 * math.log2(frequency / A4))
        
        # Simple approximation - this could be improved for Indian music
        note_index = (9 + semitones_from_A4) % 12  # A is at index 9
        octave = 4 + (9 + semitones_from_A4) // 12
        
        return f"{note_names[note_index]}{octave}"