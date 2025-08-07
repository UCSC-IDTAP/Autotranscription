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
        self.short_fixed_threshold = 0.2  # Threshold for merging short fixed segments (seconds)
    
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
                
                segments.append(segment)
        
        # Post-process to merge short fixed segments between moving segments
        segments = self._post_process_short_fixed_segments(segments, pitch_data, time_data)
        
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
    
    def segment_pitch_trace_minmax(self, pitch_data: np.ndarray, time_data: np.ndarray, 
                                  raga_object: IDTAPRaga, sample_rate: float = 44100,
                                  hop_length: int = 512) -> List[Dict[str, Any]]:
        """
        Alternative segmentation approach based on local minima and maxima.
        
        1. Find all local minima and maxima in the pitch trace
        2. Create segments between these extrema
        3. Merge adjacent segments that fall within the same target pitch threshold
        4. Label remaining segments as fixed or moving
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
        log_pitch_data = np.full_like(pitch_data, np.nan)
        valid_mask = ~np.isnan(pitch_data) & (pitch_data > 0)
        log_pitch_data[valid_mask] = np.log2(pitch_data[valid_mask])
        
        # Find silence regions first
        silence_mask = np.isnan(log_pitch_data)
        
        # Process silence and pitched regions separately
        i = 0
        while i < len(silence_mask):
            segment_start = i
            
            if silence_mask[i]:
                # Process silence segment
                while i < len(silence_mask) and silence_mask[i]:
                    i += 1
                segment_end = i
                
                segment = self._create_silence_segment(
                    segment_start, segment_end, time_data, pitch_data[segment_start:segment_end]
                )
                segments.append(segment)
            else:
                # Find end of pitched region
                while i < len(silence_mask) and not silence_mask[i]:
                    i += 1
                segment_end = i
                
                # Process this pitched region with minima/maxima approach
                pitched_segments = self._segment_pitched_region_minmax(
                    segment_start, segment_end, pitch_data, log_pitch_data, 
                    time_data, target_pitches, target_log_freqs
                )
                segments.extend(pitched_segments)
        
        return segments
    
    def _segment_pitched_region_minmax(self, start_idx: int, end_idx: int, 
                                     pitch_data: np.ndarray, log_pitch_data: np.ndarray,
                                     time_data: np.ndarray, target_pitches: List[Pitch],
                                     target_log_freqs: List[float]) -> List[Dict[str, Any]]:
        """
        Segment a pitched region using local minima and maxima.
        """
        if end_idx - start_idx < 3:
            # Too short for minima/maxima analysis, treat as single segment
            return self._create_single_segment_from_region(
                start_idx, end_idx, pitch_data, log_pitch_data, 
                time_data, target_pitches, target_log_freqs
            )
        
        region_log_pitch = log_pitch_data[start_idx:end_idx]
        valid_mask = ~np.isnan(region_log_pitch)
        
        if not np.any(valid_mask):
            return []
        
        # Find local minima and maxima
        extrema_indices = self._find_local_extrema(region_log_pitch, valid_mask)
        
        # Convert back to global indices
        extrema_indices = [idx + start_idx for idx in extrema_indices]
        
        # Add start and end points
        segment_boundaries = [start_idx] + extrema_indices + [end_idx]
        segment_boundaries = sorted(list(set(segment_boundaries)))  # Remove duplicates and sort
        
        # Create initial segments between extrema
        initial_segments = []
        for i in range(len(segment_boundaries) - 1):
            seg_start = segment_boundaries[i]
            seg_end = segment_boundaries[i + 1]
            
            if seg_end > seg_start:
                segment = self._create_single_segment_from_region(
                    seg_start, seg_end, pitch_data, log_pitch_data,
                    time_data, target_pitches, target_log_freqs
                )[0]  # This returns a list, take first element
                initial_segments.append(segment)
        
        # Merge segments that fall within the same target pitch threshold
        merged_segments = self._merge_segments_by_target_pitch(
            initial_segments, target_pitches, target_log_freqs
        )
        
        return merged_segments
    
    def _find_local_extrema(self, log_pitch_data: np.ndarray, valid_mask: np.ndarray) -> List[int]:
        """Find local minima and maxima in the pitch data."""
        extrema = []
        
        if not np.any(valid_mask):
            return extrema
        
        # Get valid data points and their indices
        valid_data = log_pitch_data[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_data) < 3:
            return extrema
        
        # Find local extrema using scipy's approach
        from scipy.signal import argrelextrema
        
        # Find local minima
        minima_idx = argrelextrema(valid_data, np.less, order=1)[0]
        # Find local maxima  
        maxima_idx = argrelextrema(valid_data, np.greater, order=1)[0]
        
        # Convert back to original indices
        minima_original = [valid_indices[idx] for idx in minima_idx]
        maxima_original = [valid_indices[idx] for idx in maxima_idx]
        
        extrema.extend(minima_original)
        extrema.extend(maxima_original)
        
        return sorted(extrema)
    
    def _create_single_segment_from_region(self, start_idx: int, end_idx: int,
                                         pitch_data: np.ndarray, log_pitch_data: np.ndarray,
                                         time_data: np.ndarray, target_pitches: List[Pitch],
                                         target_log_freqs: List[float]) -> List[Dict[str, Any]]:
        """Create a single segment from a region and determine if it's fixed or moving."""
        region_data = log_pitch_data[start_idx:end_idx]
        valid_mask = ~np.isnan(region_data)
        
        if not np.any(valid_mask):
            # No valid data, create a silence segment
            return [self._create_silence_segment(start_idx, end_idx, time_data, pitch_data[start_idx:end_idx])]
        
        valid_log_freqs = region_data[valid_mask]
        mean_log_freq = np.mean(valid_log_freqs)
        
        # Find closest target pitch
        closest_target_idx = np.argmin([abs(mean_log_freq - tlf) for tlf in target_log_freqs])
        closest_target_pitch = target_pitches[closest_target_idx]
        closest_target_log_freq = target_log_freqs[closest_target_idx]
        
        # Check if this region is stable (fixed) or moving
        freq_range = np.max(valid_log_freqs) - np.min(valid_log_freqs)
        is_stable = freq_range <= self.log_freq_threshold
        is_near_target = abs(mean_log_freq - closest_target_log_freq) <= self.log_freq_threshold
        
        if is_stable and is_near_target:
            # Fixed segment
            mean_log_offset = mean_log_freq - closest_target_log_freq
            return [self._create_fixed_pitch_segment(
                start_idx, end_idx, time_data, pitch_data[start_idx:end_idx],
                closest_target_pitch, mean_log_offset, True
            )]
        else:
            # Moving segment
            start_freq = valid_log_freqs[0]
            end_freq = valid_log_freqs[-1]
            
            start_pitch, start_in_raga = self._find_closest_pitch(start_freq, target_pitches, target_log_freqs)
            end_pitch, end_in_raga = self._find_closest_pitch(end_freq, target_pitches, target_log_freqs)
            
            return [self._create_moving_pitch_segment(
                start_idx, end_idx, time_data, pitch_data[start_idx:end_idx],
                start_pitch, end_pitch, start_in_raga and end_in_raga
            )]
    
    def _merge_segments_by_target_pitch(self, segments: List[Dict[str, Any]], 
                                      target_pitches: List[Pitch], 
                                      target_log_freqs: List[float]) -> List[Dict[str, Any]]:
        """
        Merge adjacent segments that fall within the same target pitch threshold.
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        current_group = [segments[0]]
        
        for i in range(1, len(segments)):
            current_seg = segments[i]
            prev_seg = current_group[-1]
            
            # Check if both segments are close to the same target pitch
            if self._segments_share_target_pitch(prev_seg, current_seg, target_pitches, target_log_freqs):
                current_group.append(current_seg)
            else:
                # Merge the current group and start a new group
                merged_seg = self._merge_segment_group(current_group, target_pitches, target_log_freqs)
                merged.append(merged_seg)
                current_group = [current_seg]
        
        # Don't forget the last group
        if current_group:
            merged_seg = self._merge_segment_group(current_group, target_pitches, target_log_freqs)
            merged.append(merged_seg)
        
        return merged
    
    def _segments_share_target_pitch(self, seg1: Dict[str, Any], seg2: Dict[str, Any],
                                   target_pitches: List[Pitch], target_log_freqs: List[float]) -> bool:
        """Check if two segments are close to the same target pitch."""
        # Get representative frequencies for both segments
        freq1 = self._get_segment_representative_log_frequency(seg1)
        freq2 = self._get_segment_representative_log_frequency(seg2)
        
        if freq1 == 0 or freq2 == 0:
            return False
        
        # Find closest target for each
        closest_idx1 = np.argmin([abs(freq1 - tlf) for tlf in target_log_freqs])
        closest_idx2 = np.argmin([abs(freq2 - tlf) for tlf in target_log_freqs])
        
        # Check if they share the same target and are within threshold
        same_target = closest_idx1 == closest_idx2
        within_threshold1 = abs(freq1 - target_log_freqs[closest_idx1]) <= self.log_freq_threshold
        within_threshold2 = abs(freq2 - target_log_freqs[closest_idx2]) <= self.log_freq_threshold
        
        return same_target and within_threshold1 and within_threshold2
    
    def _get_segment_representative_log_frequency(self, segment: Dict[str, Any]) -> float:
        """Get representative log frequency for a segment."""
        if 'data' not in segment:
            return 0.0
        
        data = segment['data']
        valid_mask = ~np.isnan(data) & (data > 0)
        
        if not np.any(valid_mask):
            return 0.0
        
        valid_freqs = data[valid_mask]
        return float(np.log2(np.mean(valid_freqs)))
    
    def _merge_segment_group(self, segment_group: List[Dict[str, Any]], 
                           target_pitches: List[Pitch], target_log_freqs: List[float]) -> Dict[str, Any]:
        """Merge a group of segments that share the same target pitch into one segment."""
        if len(segment_group) == 1:
            return segment_group[0]
        
        # Combine all data from the group
        first_seg = segment_group[0]
        last_seg = segment_group[-1]
        
        # Merge timing info
        merged_segment = {
            'type': 'fixed',  # Merged segments from same target are considered fixed
            'start_time': first_seg['start_time'],
            'duration': (last_seg['start_time'] + last_seg['duration']) - first_seg['start_time'],
            'in_raga': True,  # Will be determined by target pitch proximity
            '_start_idx': first_seg['_start_idx'],
            '_end_idx': last_seg['_end_idx']
        }
        
        # Combine data arrays
        all_data = []
        for seg in segment_group:
            if 'data' in seg:
                all_data.extend(seg['data'])
        merged_segment['data'] = np.array(all_data)
        
        # Determine target pitch and properties
        valid_data = [d for d in all_data if not np.isnan(d) and d > 0]
        if valid_data:
            mean_log_freq = np.log2(np.mean(valid_data))
            closest_idx = np.argmin([abs(mean_log_freq - tlf) for tlf in target_log_freqs])
            closest_target = target_pitches[closest_idx]
            closest_log_freq = target_log_freqs[closest_idx]
            
            merged_segment['pitch'] = closest_target
            merged_segment['log_offset'] = mean_log_freq - closest_log_freq
            merged_segment['in_raga'] = abs(mean_log_freq - closest_log_freq) <= self.log_freq_threshold
        
        return merged_segment
    
    def _post_process_short_fixed_segments(self, segments: List[Dict[str, Any]], 
                                         pitch_data: np.ndarray, time_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Post-process segments to merge short fixed segments between moving segments.
        
        Converts patterns like (moving, fixed<0.2s, moving) -> (moving, moving)
        """
        if len(segments) < 3:
            return segments
            
        processed_segments = []
        i = 0
        
        while i < len(segments):
            current_seg = segments[i]
            
            # Check if this is a short fixed segment between two moving segments
            if (i > 0 and i < len(segments) - 1 and 
                current_seg['type'] == 'fixed' and 
                current_seg['duration'] < self.short_fixed_threshold and
                segments[i-1]['type'] == 'moving' and 
                segments[i+1]['type'] == 'moving'):
                
                # Get the previous moving segment (already in processed_segments)
                prev_moving = processed_segments[-1]
                next_moving = segments[i+1]
                
                # Merge the short fixed segment with adjacent moving segments
                merged_segments = self._merge_short_fixed_segment(
                    prev_moving, current_seg, next_moving, pitch_data, time_data
                )
                
                # Replace the last processed segment with the first merged segment
                processed_segments[-1] = merged_segments[0]
                # Add the second merged segment
                processed_segments.append(merged_segments[1])
                
                # Skip the next moving segment since we've already processed it
                i += 2
            else:
                processed_segments.append(current_seg)
                i += 1
                
        return processed_segments
    
    def _merge_short_fixed_segment(self, prev_moving: Dict[str, Any], fixed_seg: Dict[str, Any], 
                                 next_moving: Dict[str, Any], pitch_data: np.ndarray, 
                                 time_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Merge a short fixed segment with adjacent moving segments using frequency-based division rules.
        
        Rules:
        1. Both moving segments higher than fixed: divide at lowest frequency point
        2. Both moving segments lower than fixed: divide at highest frequency point  
        3. One above, one below: divide where it crosses target pitch
        """
        # Get frequency data for the fixed segment
        fixed_start_idx = fixed_seg['_start_idx']
        fixed_end_idx = fixed_seg['_end_idx']
        fixed_pitch_data = pitch_data[fixed_start_idx:fixed_end_idx]
        fixed_time_data = time_data[fixed_start_idx:fixed_end_idx]
        
        # Remove NaN/zero values
        valid_mask = ~np.isnan(fixed_pitch_data) & (fixed_pitch_data > 0)
        if not np.any(valid_mask):
            # If no valid data, just extend the moving segments to meet in the middle
            mid_idx = fixed_start_idx + (fixed_end_idx - fixed_start_idx) // 2
            division_point = mid_idx
        else:
            valid_freqs = fixed_pitch_data[valid_mask]
            valid_indices = np.where(valid_mask)[0] + fixed_start_idx
            
            # Get representative frequencies for the moving segments
            prev_freq = self._get_segment_representative_frequency(prev_moving, pitch_data)
            next_freq = self._get_segment_representative_frequency(next_moving, pitch_data)
            fixed_target_freq = fixed_seg['pitch'].frequency
            
            # Apply division rules
            if prev_freq > fixed_target_freq and next_freq > fixed_target_freq:
                # Rule 1: Both higher - divide at lowest point
                min_idx = np.argmin(valid_freqs)
                division_point = valid_indices[min_idx]
            elif prev_freq < fixed_target_freq and next_freq < fixed_target_freq:
                # Rule 2: Both lower - divide at highest point
                max_idx = np.argmax(valid_freqs)
                division_point = valid_indices[max_idx]
            else:
                # Rule 3: One above, one below - divide at target crossing
                # Find point closest to target frequency
                freq_diffs = np.abs(valid_freqs - fixed_target_freq)
                closest_idx = np.argmin(freq_diffs)
                division_point = valid_indices[closest_idx]
        
        # Create two new moving segments
        # First moving segment: from prev_moving start to division point
        first_end_idx = division_point
        first_moving = prev_moving.copy()
        first_moving['_end_idx'] = first_end_idx
        first_moving['duration'] = time_data[first_end_idx] - first_moving['start_time']
        
        # Second moving segment: from division point to next_moving end  
        second_start_idx = division_point
        second_moving = next_moving.copy()
        second_moving['_start_idx'] = second_start_idx
        second_moving['start_time'] = time_data[second_start_idx]
        second_moving['duration'] = second_moving['start_time'] + second_moving['duration'] - time_data[second_start_idx]
        
        return [first_moving, second_moving]
    
    def _get_segment_representative_frequency(self, segment: Dict[str, Any], pitch_data: np.ndarray) -> float:
        """Get a representative frequency for a segment (mean of valid frequencies)."""
        start_idx = segment['_start_idx']
        end_idx = segment['_end_idx']
        seg_data = pitch_data[start_idx:end_idx]
        
        # Remove NaN/zero values
        valid_mask = ~np.isnan(seg_data) & (seg_data > 0)
        if not np.any(valid_mask):
            return 0.0
            
        return np.mean(seg_data[valid_mask])


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