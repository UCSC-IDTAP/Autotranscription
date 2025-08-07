"""
Pitch cleaning module for auto-transcription pipeline.

Handles post-processing cleaning of pitch traces to remove octave jumps and other artifacts.
Designed to work after pitch extraction to clean short-duration octave errors.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import json


class PitchCleaner:
    """
    Post-processing pitch trace cleaner for removing octave jumps and artifacts.
    
    Focuses on correcting short-duration octave errors that slip through
    the initial pitch extraction algorithms.
    """
    
    def __init__(self, 
                 octave_ratio_tolerance: float = 0.1,
                 max_jump_duration_sec: float = 0.25,
                 context_frames: int = 5,
                 confidence_threshold: float = 0.05):
        """
        Initialize the pitch cleaner.
        
        Args:
            octave_ratio_tolerance: Tolerance for detecting octave ratios (default: 0.1)
            max_jump_duration_sec: Maximum duration of jumps to correct in seconds (default: 0.25)
            context_frames: Number of frames before/after to use for context (default: 5)
            confidence_threshold: Minimum confidence to process a frame (default: 0.05)
        """
        self.octave_ratio_tolerance = octave_ratio_tolerance
        self.max_jump_duration_sec = max_jump_duration_sec
        self.context_frames = context_frames
        self.confidence_threshold = confidence_threshold
        
        # Define octave ratio ranges
        self.octave_up_min = 2.0 - octave_ratio_tolerance
        self.octave_up_max = 2.0 + octave_ratio_tolerance
        self.octave_down_min = 0.5 - octave_ratio_tolerance
        self.octave_down_max = 0.5 + octave_ratio_tolerance
    
    def clean_pitch_trace(self, 
                         pitch_values: np.ndarray, 
                         confidence_values: np.ndarray, 
                         times: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Clean pitch trace by correcting short-duration octave jumps.
        
        Args:
            pitch_values: Array of pitch values in Hz
            confidence_values: Array of confidence values
            times: Array of time values in seconds
            
        Returns:
            Tuple of (cleaned_pitch_values, cleaning_stats)
        """
        # Create copies to avoid modifying originals
        cleaned_pitch = pitch_values.copy()
        stats = {
            'total_frames': len(pitch_values),
            'octave_jumps_detected': 0,
            'octave_jumps_corrected': 0,
            'corrections_applied': []
        }
        
        # Calculate sample rate from times
        if len(times) > 1:
            sample_rate_hz = 1.0 / (times[1] - times[0])
            max_jump_frames = int(self.max_jump_duration_sec * sample_rate_hz)
        else:
            max_jump_frames = 25  # Default fallback for ~0.25s at 100Hz
        
        # Find all octave jumps
        octave_jumps = self._detect_octave_jumps(pitch_values, confidence_values)
        stats['octave_jumps_detected'] = len(octave_jumps)
        
        # Process each octave jump for potential correction
        for jump_info in octave_jumps:
            correction = self._analyze_and_correct_jump(
                cleaned_pitch, confidence_values, times, jump_info, max_jump_frames
            )
            
            if correction is not None:
                start_frame, end_frame, correction_factor = correction
                
                # Apply correction
                for i in range(start_frame, end_frame + 1):
                    if cleaned_pitch[i] > 0:  # Only correct voiced frames
                        cleaned_pitch[i] *= correction_factor
                
                stats['octave_jumps_corrected'] += 1
                stats['corrections_applied'].append({
                    'start_frame': int(start_frame),
                    'end_frame': int(end_frame),
                    'start_time': float(times[start_frame]) if start_frame < len(times) else 0.0,
                    'end_time': float(times[end_frame]) if end_frame < len(times) else 0.0,
                    'correction_factor': float(correction_factor),
                    'duration_frames': int(end_frame - start_frame + 1)
                })
        
        return cleaned_pitch, stats
    
    def _detect_octave_jumps(self, pitch_values: np.ndarray, 
                           confidence_values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect all octave jumps in the pitch sequence.
        
        Args:
            pitch_values: Array of pitch values in Hz
            confidence_values: Array of confidence values
            
        Returns:
            List of octave jump information dictionaries
        """
        jumps = []
        
        for i in range(1, len(pitch_values)):
            # Only analyze frames with sufficient confidence
            if (confidence_values[i] < self.confidence_threshold or 
                confidence_values[i-1] < self.confidence_threshold):
                continue
                
            # Only analyze voiced frames (pitch > 0)
            if pitch_values[i] <= 0 or pitch_values[i-1] <= 0:
                continue
            
            ratio = pitch_values[i] / pitch_values[i-1]
            
            # Check for octave jumps
            is_octave_up = self.octave_up_min <= ratio <= self.octave_up_max
            is_octave_down = self.octave_down_min <= ratio <= self.octave_down_max
            
            if is_octave_up or is_octave_down:
                jumps.append({
                    'frame': i,
                    'ratio': ratio,
                    'direction': 'up' if is_octave_up else 'down',
                    'freq_before': pitch_values[i-1],
                    'freq_after': pitch_values[i]
                })
        
        return jumps
    
    def _analyze_and_correct_jump(self, 
                                 pitch_values: np.ndarray,
                                 confidence_values: np.ndarray,
                                 times: np.ndarray,
                                 jump_info: Dict[str, Any],
                                 max_jump_frames: int) -> Optional[Tuple[int, int, float]]:
        """
        Analyze an octave jump and determine if it should be corrected.
        
        Args:
            pitch_values: Array of pitch values
            confidence_values: Array of confidence values
            times: Array of time values
            jump_info: Information about the octave jump
            max_jump_frames: Maximum number of frames for a correctable jump
            
        Returns:
            Tuple of (start_frame, end_frame, correction_factor) if correction should be applied,
            None otherwise
        """
        jump_frame = jump_info['frame']
        jump_direction = jump_info['direction']
        
        # Find the extent of the octave-shifted region
        shifted_start = jump_frame
        shifted_end = jump_frame
        
        # Extend forward to find where the octave shift ends
        target_freq = jump_info['freq_after']
        for i in range(jump_frame + 1, min(jump_frame + max_jump_frames, len(pitch_values))):
            if pitch_values[i] <= 0 or confidence_values[i] < self.confidence_threshold:
                break
            
            # Check if we're still in the octave-shifted region
            if jump_direction == 'up':
                # For upward jumps, look for frequencies in the upper octave
                if pitch_values[i] > target_freq * 0.8 and pitch_values[i] < target_freq * 1.25:
                    shifted_end = i
                else:
                    break
            else:
                # For downward jumps, look for frequencies in the lower octave
                if pitch_values[i] > target_freq * 0.8 and pitch_values[i] < target_freq * 1.25:
                    shifted_end = i
                else:
                    break
        
        # Calculate duration of the shifted region
        shift_duration = shifted_end - shifted_start + 1
        
        # Only correct short-duration shifts
        if shift_duration > max_jump_frames:
            return None
        
        # Check context before and after the shift
        context_before = self._get_context_frequency(
            pitch_values, confidence_values, shifted_start - 1, -1
        )
        context_after = self._get_context_frequency(
            pitch_values, confidence_values, shifted_end + 1, 1
        )
        
        if context_before is None and context_after is None:
            return None
        
        # Determine the correct octave based on context
        if context_before is not None and context_after is not None:
            # Use both contexts - prefer the one with smaller correction
            target_freq_before = context_before
            target_freq_after = context_after
            
            # Choose context that requires smaller correction
            correction_before = target_freq_before / jump_info['freq_after']
            correction_after = target_freq_after / jump_info['freq_after']
            
            if abs(np.log2(correction_before)) < abs(np.log2(correction_after)):
                target_freq = target_freq_before
            else:
                target_freq = target_freq_after
        elif context_before is not None:
            target_freq = context_before
        else:
            target_freq = context_after
        
        # Calculate correction factor
        correction_factor = target_freq / jump_info['freq_after']
        
        # Only apply correction if it's approximately an octave correction
        correction_ratio = abs(np.log2(correction_factor))
        if 0.9 <= correction_ratio <= 1.1:  # Close to 1 octave (2^1 = 2)
            return shifted_start, shifted_end, correction_factor
        
        return None
    
    def _get_context_frequency(self, 
                              pitch_values: np.ndarray,
                              confidence_values: np.ndarray,
                              start_frame: int,
                              direction: int) -> Optional[float]:
        """
        Get representative frequency from context frames.
        
        Args:
            pitch_values: Array of pitch values
            confidence_values: Array of confidence values
            start_frame: Starting frame for context analysis
            direction: Direction to search (-1 for backward, 1 for forward)
            
        Returns:
            Representative frequency from context, or None if no valid context
        """
        valid_frequencies = []
        
        for offset in range(1, self.context_frames + 1):
            frame = start_frame + (offset * direction)
            
            if frame < 0 or frame >= len(pitch_values):
                break
                
            if (pitch_values[frame] > 0 and 
                confidence_values[frame] >= self.confidence_threshold):
                valid_frequencies.append(pitch_values[frame])
        
        if len(valid_frequencies) == 0:
            return None
        
        # Return median frequency as representative
        return float(np.median(valid_frequencies))
    
    def save_cleaning_results(self, 
                            audio_id: int, 
                            algorithm: str,
                            cleaned_pitch: np.ndarray,
                            cleaning_stats: Dict[str, Any],
                            output_dir: Path) -> Dict[str, Any]:
        """
        Save cleaned pitch data and statistics.
        
        Args:
            audio_id: Unique identifier for the audio file
            algorithm: Name of the pitch extraction algorithm
            cleaned_pitch: Cleaned pitch values
            cleaning_stats: Cleaning statistics
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths and cleaning information
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned pitch data
        cleaned_file = output_dir / f"{audio_id}_{algorithm}_cleaned.npy"
        np.save(cleaned_file, cleaned_pitch)
        
        # Save cleaning statistics
        stats_file = output_dir / f"{audio_id}_{algorithm}_cleaning_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(cleaning_stats, f, indent=2)
        
        return {
            'cleaned_pitch_file': str(cleaned_file),
            'cleaning_stats_file': str(stats_file),
            'octave_jumps_detected': cleaning_stats['octave_jumps_detected'],
            'octave_jumps_corrected': cleaning_stats['octave_jumps_corrected'],
            'corrections_applied': len(cleaning_stats['corrections_applied'])
        }


def clean_pitch_data(pitch_values: np.ndarray, 
                    confidence_values: np.ndarray, 
                    times: np.ndarray,
                    **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function for cleaning pitch data.
    
    Args:
        pitch_values: Array of pitch values in Hz
        confidence_values: Array of confidence values
        times: Array of time values in seconds
        **kwargs: Additional parameters for PitchCleaner
        
    Returns:
        Tuple of (cleaned_pitch_values, cleaning_stats)
    """
    cleaner = PitchCleaner(**kwargs)
    return cleaner.clean_pitch_trace(pitch_values, confidence_values, times)


if __name__ == "__main__":
    # Test the pitch cleaner with existing data
    import sys
    from pathlib import Path
    
    # Load test data
    workspace_dir = Path("/Users/jon/Documents/2025/idtap_project/Autotranscription/pipeline_workspace")
    pitch_file = workspace_dir / "data" / "pitch_traces" / "0_vocals_essentia.npy"
    confidence_file = workspace_dir / "data" / "pitch_traces" / "0_vocals_essentia_confidence.npy"
    times_file = workspace_dir / "data" / "pitch_traces" / "0_vocals_essentia_times.npy"
    
    if all(f.exists() for f in [pitch_file, confidence_file, times_file]):
        print("Loading test data...")
        pitch_data = np.load(pitch_file)
        confidence_data = np.load(confidence_file)
        times_data = np.load(times_file)
        
        print(f"Original data: {len(pitch_data)} frames")
        print(f"Frequency range: {pitch_data[pitch_data > 0].min():.1f} - {pitch_data[pitch_data > 0].max():.1f} Hz")
        
        # Clean the data
        cleaner = PitchCleaner()
        cleaned_pitch, stats = cleaner.clean_pitch_trace(pitch_data, confidence_data, times_data)
        
        print(f"\nCleaning results:")
        print(f"  Octave jumps detected: {stats['octave_jumps_detected']}")
        print(f"  Octave jumps corrected: {stats['octave_jumps_corrected']}")
        print(f"  Corrections applied: {len(stats['corrections_applied'])}")
        
        if stats['corrections_applied']:
            print("\nCorrections applied:")
            for correction in stats['corrections_applied']:
                print(f"  Frames {correction['start_frame']}-{correction['end_frame']}: "
                      f"factor {correction['correction_factor']:.3f} "
                      f"({correction['duration_frames']} frames)")
    else:
        print("Test data files not found. Run the pipeline first to generate test data.")