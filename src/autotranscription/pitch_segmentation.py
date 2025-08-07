"""
Pitch segmentation module for auto-transcription pipeline.

Segments pitch traces into fixed pitch, moving pitch, and silence segments
with support for out-of-raga pitch detection.
"""

import numpy as np
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    # Add the Python-API path to handle the installed idtap package
    import sys
    sys.path.insert(0, '/Users/jon/documents/2025/idtap_project/Python-API')
    from idtap.classes.raga import Raga as IDTAPRaga
    from idtap.classes.pitch import Pitch
except ImportError:
    print("Warning: idtap not available. Pitch segmentation will use fallback methods.")
    IDTAPRaga = None
    Pitch = None


class PitchSegmenter:
    """
    Segments pitch traces into meaningful musical units for Indian classical music.
    
    Segments are classified as:
    1. Fixed pitch: regions stable within ±1/24 log frequency units of raga pitches
    2. Moving pitch: transitions between pitches
    3. Silence: regions with no valid pitch data
    """
    
    def __init__(self, min_duration: float = 0.25):
        """
        Initialize the pitch segmenter.
        
        Args:
            min_duration: Minimum duration for segments in seconds (default: 0.05s)
        """
        self.min_duration = min_duration
        self.log_freq_threshold = 1/24  # ±1/24 units (half semitone, ~50 cents)
    
    def segment_pitch_trace(self, pitch_data: np.ndarray, time_data: np.ndarray, 
                          raga_object: IDTAPRaga, sample_rate: float = 44100,
                          hop_length: int = 512) -> List[Dict[str, Any]]:
        """
        Segment pitch trace into fixed, moving, and silence segments.
        
        Args:
            pitch_data: Array of pitch frequencies (Hz), NaN for silence
            time_data: Array of time values corresponding to pitch_data
            raga_object: IDTAP Raga object for target pitch calculations
            sample_rate: Audio sample rate
            hop_length: Hop length used in pitch extraction
            
        Returns:
            List of segment dictionaries with type, timing, pitch info, and data
        """
        if IDTAPRaga is None or Pitch is None:
            raise ImportError("IDTAP API not available for pitch segmentation")
        
        segments = []
        
        # Get target raga pitches in the frequency range
        min_freq = np.nanmin(pitch_data) if not np.isnan(np.nanmin(pitch_data)) else 75
        max_freq = np.nanmax(pitch_data) if not np.isnan(np.nanmax(pitch_data)) else 800
        target_pitches = raga_object.get_pitches(low=max(min_freq*0.8, 75), 
                                                high=min(max_freq*1.2, 2400))
        
        # Convert to log frequencies for comparison
        target_log_freqs = [p.non_offset_log_freq for p in target_pitches]
        
        # Convert pitch data to log frequencies
        # Treat 0.0 values as silence (many pitch extractors use 0.0 to indicate no pitch)
        log_pitch_data = np.full_like(pitch_data, np.nan)
        valid_mask = ~np.isnan(pitch_data) & (pitch_data > 0)
        log_pitch_data[valid_mask] = np.log2(pitch_data[valid_mask])
        
        # Find silence regions
        silence_mask = np.isnan(log_pitch_data)
        
        # Create fine-grained segments by analyzing pitch changes more sensitively
        segments = []
        i = 0
        
        while i < len(log_pitch_data):
            segment_start = i
            
            if silence_mask[i]:
                # Process silence segment
                while i < len(silence_mask) and silence_mask[i]:
                    i += 1
                segment_end = i
                
                segment = self._create_silence_segment(
                    segment_start, segment_end, time_data, pitch_data[segment_start:segment_end]
                )
                if segment['duration'] >= self.min_duration:
                    segments.append(segment)
                    
            else:
                # For pitched segments, create shorter segments based on pitch stability
                current_log_freq = log_pitch_data[i]
                
                # Find closest target pitch for reference
                closest_target_idx = np.argmin([abs(current_log_freq - tlf) for tlf in target_log_freqs])
                target_log_freq = target_log_freqs[closest_target_idx]
                target_pitch = target_pitches[closest_target_idx]
                
                # Check if this is close to a raga pitch (fixed) or not (moving)
                is_near_raga = abs(current_log_freq - target_log_freq) <= self.log_freq_threshold
                
                if is_near_raga:
                    # Fixed pitch segment - extend while close to this target
                    while (i < len(log_pitch_data) and 
                           not silence_mask[i] and 
                           abs(log_pitch_data[i] - target_log_freq) <= self.log_freq_threshold):
                        i += 1
                    segment_end = i
                    
                    region_data = log_pitch_data[segment_start:segment_end]
                    mean_log_offset = np.mean(region_data - target_log_freq)
                    
                    segment = self._create_fixed_pitch_segment(
                        segment_start, segment_end, time_data, 
                        pitch_data[segment_start:segment_end],
                        target_pitch, mean_log_offset, True
                    )
                else:
                    # Moving pitch segment - create shorter segments by detecting direction changes
                    start_pitch, start_in_raga = self._find_closest_pitch(
                        current_log_freq, target_pitches, target_log_freqs
                    )
                    
                    # Look for pitch direction changes to create finer segments
                    prev_log_freq = current_log_freq
                    direction = 0  # 0=stable, 1=up, -1=down
                    direction_changes = 0
                    max_segment_length = min(50, len(log_pitch_data) - i)  # Max 50 frames per segment
                    
                    # Advance through the moving segment
                    segment_length = 1
                    while (i + segment_length < len(log_pitch_data) and 
                           not silence_mask[i + segment_length] and 
                           segment_length < max_segment_length):
                        
                        next_log_freq = log_pitch_data[i + segment_length]
                        
                        # Check if we're approaching a raga pitch (might become fixed)
                        min_distance_to_raga = min([abs(next_log_freq - tlf) for tlf in target_log_freqs])
                        if min_distance_to_raga <= self.log_freq_threshold:
                            # Stop here - next segment might be fixed
                            break
                        
                        # Detect direction changes for finer segmentation
                        freq_diff = next_log_freq - prev_log_freq
                        if abs(freq_diff) > 0.002:  # Small threshold for direction
                            new_direction = 1 if freq_diff > 0 else -1
                            if direction != 0 and direction != new_direction:
                                direction_changes += 1
                                if direction_changes >= 2:  # After a few direction changes, start new segment
                                    break
                            direction = new_direction
                        
                        prev_log_freq = next_log_freq
                        segment_length += 1
                    
                    segment_end = i + segment_length
                    i = segment_end
                    
                    # Get end pitch
                    end_log_freq = log_pitch_data[segment_end - 1]
                    end_pitch, end_in_raga = self._find_closest_pitch(
                        end_log_freq, target_pitches, target_log_freqs
                    )
                    
                    segment = self._create_moving_pitch_segment(
                        segment_start, segment_end, time_data,
                        pitch_data[segment_start:segment_end],
                        start_pitch, end_pitch, start_in_raga and end_in_raga
                    )
                
                if segment['duration'] >= self.min_duration:
                    segments.append(segment)
        
        return segments
    
    def _find_closest_pitch(self, log_freq: float, target_pitches: List[Pitch],
                           target_log_freqs: List[float]) -> Tuple[Pitch, bool]:
        """
        Find closest pitch to given log frequency and determine if it's in raga.
        """
        closest_idx = np.argmin([abs(log_freq - tlf) for tlf in target_log_freqs])
        closest_pitch = target_pitches[closest_idx]
        closest_log_freq = target_log_freqs[closest_idx]
        
        # Check if within raga threshold
        in_raga = abs(log_freq - closest_log_freq) <= self.log_freq_threshold
        
        if not in_raga:
            # Create out-of-raga pitch by finding the closest chromatic pitch
            # and adding appropriate log_offset
            log_offset = log_freq - closest_log_freq
            out_of_raga_pitch = Pitch({
                'swara': closest_pitch.swara,
                'oct': closest_pitch.oct,
                'raised': closest_pitch.raised,
                'fundamental': closest_pitch.fundamental,
                'ratios': closest_pitch.ratios,
                'log_offset': log_offset
            })
            return out_of_raga_pitch, False
        
        return closest_pitch, True
    
    def _create_fixed_pitch_segment(self, start_idx: int, end_idx: int, time_data: np.ndarray,
                                  pitch_data: np.ndarray, pitch: Pitch, log_offset: float,
                                  in_raga: bool) -> Dict[str, Any]:
        """Create a fixed pitch segment dictionary."""
        start_time = time_data[start_idx]
        end_time = time_data[end_idx - 1] if end_idx > start_idx else time_data[start_idx]
        
        return {
            'type': 'fixed',
            'start_time': float(start_time),
            'duration': float(end_time - start_time),
            'pitch': pitch,
            'log_offset': log_offset,
            'data': pitch_data.copy(),
            'in_raga': in_raga,
            '_start_idx': start_idx,
            '_end_idx': end_idx
        }
    
    def _create_moving_pitch_segment(self, start_idx: int, end_idx: int, time_data: np.ndarray,
                                   pitch_data: np.ndarray, start_pitch: Pitch, end_pitch: Pitch,
                                   in_raga: bool) -> Dict[str, Any]:
        """Create a moving pitch segment dictionary."""
        start_time = time_data[start_idx]
        end_time = time_data[end_idx - 1] if end_idx > start_idx else time_data[start_idx]
        
        return {
            'type': 'moving',
            'start_time': float(start_time),
            'duration': float(end_time - start_time),
            'start_pitch': start_pitch,
            'end_pitch': end_pitch,
            'data': pitch_data.copy(),
            'in_raga': in_raga,
            '_start_idx': start_idx,
            '_end_idx': end_idx
        }
    
    def _create_silence_segment(self, start_idx: int, end_idx: int, time_data: np.ndarray,
                              pitch_data: np.ndarray) -> Dict[str, Any]:
        """Create a silence segment dictionary."""
        start_time = time_data[start_idx]
        end_time = time_data[end_idx - 1] if end_idx > start_idx else time_data[start_idx]
        
        return {
            'type': 'silence',
            'start_time': float(start_time),
            'duration': float(end_time - start_time),
            'data': pitch_data[start_idx:end_idx].copy(),
            '_start_idx': start_idx,
            '_end_idx': end_idx
        }
    
    def segment_from_pipeline_data(self, audio_id: int, workspace_dir: Path, 
                                 raga_object: IDTAPRaga) -> List[Dict[str, Any]]:
        """
        Load pitch trace data from pipeline and perform segmentation.
        
        Args:
            audio_id: Audio file identifier
            workspace_dir: Pipeline workspace directory
            raga_object: IDTAP Raga object for the piece
            
        Returns:
            List of segment dictionaries
        """
        # Load pitch trace data
        pitch_traces_dir = workspace_dir / "data" / "pitch_traces"
        pitch_file = pitch_traces_dir / f"{audio_id}_vocals_essentia.npy"
        confidence_file = pitch_traces_dir / f"{audio_id}_vocals_essentia_confidence.npy"
        times_file = pitch_traces_dir / f"{audio_id}_vocals_essentia_times.npy"
        
        if not pitch_file.exists():
            raise FileNotFoundError(f"Pitch trace file not found: {pitch_file}")
        if not confidence_file.exists():
            raise FileNotFoundError(f"Confidence file not found: {confidence_file}")
        
        pitch_data = np.load(pitch_file)
        confidence_data = np.load(confidence_file)
        
        # Load time data if available, otherwise create it
        if times_file.exists():
            time_data = np.load(times_file)
        else:
            # Create time array (assuming 44.1kHz sample rate, 512 hop length)
            hop_length = 512
            sample_rate = 44100
            time_data = np.arange(len(pitch_data)) * hop_length / sample_rate
        
        # Apply confidence filtering (use Essentia threshold of 0.05)
        confidence_threshold = 0.05
        low_confidence_mask = confidence_data <= confidence_threshold
        
        # Set low-confidence pitch values to 0.0 (will be treated as silence)
        filtered_pitch_data = pitch_data.copy()
        filtered_pitch_data[low_confidence_mask] = 0.0
        
        return self.segment_pitch_trace(filtered_pitch_data, time_data, raga_object)


def segment_pitch_trace_pipeline(audio_id: int, workspace_dir: Union[str, Path],
                               raga_object: IDTAPRaga, min_duration: float = 0.25) -> List[Dict[str, Any]]:
    """
    Convenience function to segment pitch traces from pipeline data.
    
    Args:
        audio_id: Audio file identifier
        workspace_dir: Pipeline workspace directory  
        raga_object: IDTAP Raga object for the piece
        min_duration: Minimum segment duration in seconds
        
    Returns:
        List of segment dictionaries
    """
    workspace_path = Path(workspace_dir)
    segmenter = PitchSegmenter(min_duration=min_duration)
    return segmenter.segment_from_pipeline_data(audio_id, workspace_path, raga_object)