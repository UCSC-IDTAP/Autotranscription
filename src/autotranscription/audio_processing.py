"""
Audio processing utilities for auto-transcription pipeline.

Contains functions for audio manipulation, envelope extraction, and sine wave generation.
"""

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from typing import Dict, Any


class AudioProcessor:
    """
    Handles audio processing utilities for the transcription pipeline.
    """
    
    def create_pitch_audio_outputs(self, audio_id: int, vocals_path: Path, 
                                 pitch_results: Dict[str, Any], 
                                 audio_outputs_dir: Path,
                                 confidence_thresholds: Dict[str, float]):
        """
        Create sine wave audio files for all pitch traces.
        
        Args:
            audio_id: Unique identifier for the audio file
            vocals_path: Path to vocals audio file
            pitch_results: Pitch trace extraction results
            audio_outputs_dir: Directory to save audio output files
            confidence_thresholds: Algorithm-specific confidence thresholds
        """
        # Load vocals audio for reference length and sample rate
        vocals_sample_rate, vocals_audio = wavfile.read(vocals_path)
        
        # Ensure mono
        if len(vocals_audio.shape) > 1:
            vocals_audio = vocals_audio.mean(axis=1)
        
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
            if key in pitch_results:
                # Load pitch trace data
                pitch_file = pitch_results[key]['pitch_data_file']
                confidence_file = pitch_results[key]['confidence_data_file']
                times_file = pitch_results[key]['times_data_file']
                
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
                sine_file = audio_outputs_dir / f"{audio_id}_{filename_suffix}_pitch_sine.wav"
                sine_int = (sine_wave_with_envelope * 32767).astype(np.int16)
                wavfile.write(sine_file, vocals_sample_rate, sine_int)
    
    def _generate_sine_overlay(self, pitch_values: np.ndarray, valid_mask: np.ndarray, 
                             times: np.ndarray, audio_length: int, sample_rate: int) -> np.ndarray:
        """
        Generate a sine wave overlay based on pitch trace data with continuous frequency modulation.
        
        Args:
            pitch_values: Array of pitch frequencies
            valid_mask: Boolean mask for valid pitch points
            times: Time axis for pitch values
            audio_length: Length of output audio in samples
            sample_rate: Audio sample rate
            
        Returns:
            Generated sine wave array
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
        
        Args:
            audio: Input audio signal
            sample_rate: Audio sample rate
            
        Returns:
            Amplitude envelope normalized to 0-1 range
        """
        # Use Hilbert transform to get the analytic signal
        analytic_signal = hilbert(audio)
        
        # Get the amplitude envelope (magnitude of analytic signal)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Smooth the envelope to reduce noise
        # Apply a simple moving average filter
        window_size = int(0.01 * sample_rate)  # 10ms window
        if window_size > 1:
            amplitude_envelope = uniform_filter1d(amplitude_envelope.astype(np.float64), size=window_size)
        
        # Normalize to 0-1 range
        if amplitude_envelope.max() > 0:
            amplitude_envelope = amplitude_envelope / amplitude_envelope.max()
        
        return amplitude_envelope