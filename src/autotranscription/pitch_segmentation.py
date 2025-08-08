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
        # Allow lenient fixed classification when most frames lie on the same target
        self.fixed_majority_ratio = 0.9  # 90% of frames within threshold counts as fixed
    
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
        # Ensure minimum segment duration by using proper end index
        actual_end_idx = max(end_idx - 1, start_idx)
        end_time = time_data[actual_end_idx] if actual_end_idx < len(time_data) else time_data[-1]
        
        # For single-sample segments, ensure minimum hop duration
        if end_time == start_time and actual_end_idx + 1 < len(time_data):
            end_time = time_data[actual_end_idx + 1]
        
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
        # Ensure minimum segment duration by using proper end index
        actual_end_idx = max(end_idx - 1, start_idx)
        end_time = time_data[actual_end_idx] if actual_end_idx < len(time_data) else time_data[-1]
        
        # For single-sample segments, ensure minimum hop duration
        if end_time == start_time and actual_end_idx + 1 < len(time_data):
            end_time = time_data[actual_end_idx + 1]
        
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
        # Ensure minimum segment duration by using proper end index
        actual_end_idx = max(end_idx - 1, start_idx)
        end_time = time_data[actual_end_idx] if actual_end_idx < len(time_data) else time_data[-1]
        
        # For single-sample segments, ensure minimum hop duration
        if end_time == start_time and actual_end_idx + 1 < len(time_data):
            end_time = time_data[actual_end_idx + 1]
        
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
                                  hop_length: int = 512) -> Dict[str, Any]:
        """
        Two-stage segmentation approach based on local minima and maxima.
        
        Stage 1: Find very local minima and maxima to create fine-grained segments
        Stage 2: Merge consecutive segments that all fall within 1/24th octave threshold
        
        Returns both stage results for visualization.
        """
        if IDTAPRaga is None or Pitch is None:
            raise ImportError("IDTAP API not available for pitch segmentation")
        
        # Stage 1: Fine-grained segmentation at every local min/max
        stage1_segments = self._stage1_fine_segmentation(pitch_data, time_data, raga_object)
        
        # Stage 2: Merge segments within threshold
        stage2_segments = self._stage2_merge_within_threshold(stage1_segments, raga_object)
        
        return {
            'stage1_segments': stage1_segments,
            'stage2_segments': stage2_segments,
            'final_segments': stage2_segments  # For backward compatibility
        }
    
    def _stage1_fine_segmentation(self, pitch_data: np.ndarray, time_data: np.ndarray, 
                                 raga_object: IDTAPRaga) -> List[Dict[str, Any]]:
        """
        Stage 1: Create fine-grained segments at every local minimum and maximum.
        Uses more local detection than the original method.
        """
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
                
                # Process this pitched region with very fine min/max approach
                pitched_segments = self._segment_pitched_region_very_fine(
                    segment_start, segment_end, pitch_data, log_pitch_data, 
                    time_data, target_pitches, target_log_freqs
                )
                segments.extend(pitched_segments)
        
        return segments
    
    def _stage2_merge_within_threshold(self, stage1_segments: List[Dict[str, Any]], 
                                     raga_object: IDTAPRaga) -> List[Dict[str, Any]]:
        """
        Stage 2: First detect vibrato segments, then merge consecutive 'fixed' segments.
        
        Steps:
        1. Detect vibrato sequences (rapid oscillations between adjacent pitches)
        2. Merge consecutive segments with same target pitch
        """
        if len(stage1_segments) <= 1:
            return stage1_segments
        
        # Step 1: Detect vibrato segments first
        segments_after_vibrato = self._detect_vibrato_segments(stage1_segments, raga_object)
        
        # Step 2: Continue with normal merging on remaining segments
        
        # Get target pitches for comparison
        pitch_data_all = np.concatenate([seg['data'] for seg in segments_after_vibrato if 'data' in seg])
        min_freq = np.nanmin(pitch_data_all) if len(pitch_data_all) > 0 and not np.isnan(np.nanmin(pitch_data_all)) else 75
        max_freq = np.nanmax(pitch_data_all) if len(pitch_data_all) > 0 and not np.isnan(np.nanmax(pitch_data_all)) else 800
        target_pitches = raga_object.get_pitches(low=max(min_freq*0.8, 75), 
                                                high=min(max_freq*1.2, 2400))
        target_log_freqs = [p.non_offset_log_freq for p in target_pitches]
        
        merged_segments = []
        current_fixed_group = []
        current_target_pitch_idx = None
        
        for segment in segments_after_vibrato:
            if segment['type'] == 'silence':
                # Process any current fixed group first
                if current_fixed_group:
                    merged_seg = self._merge_fixed_segment_group(current_fixed_group, target_pitches, target_log_freqs)
                    merged_segments.append(merged_seg)
                    current_fixed_group = []
                    current_target_pitch_idx = None
                
                # Add silence segment as-is
                merged_segments.append(segment)
                continue
            
            if segment['type'] == 'vibrato':
                # Process any current fixed group first
                if current_fixed_group:
                    merged_seg = self._merge_fixed_segment_group(current_fixed_group, target_pitches, target_log_freqs)
                    merged_segments.append(merged_seg)
                    current_fixed_group = []
                    current_target_pitch_idx = None
                
                # Add vibrato segment as-is (already processed)
                merged_segments.append(segment)
                continue
            
            if segment['type'] == 'fixed':
                # Determine which target pitch this fixed segment corresponds to
                seg_target_idx = self._get_segment_target_pitch_index(segment, target_log_freqs)
                
                # Check if we can merge with the current group
                if (current_target_pitch_idx is not None and 
                    seg_target_idx == current_target_pitch_idx and
                    len(current_fixed_group) > 0):
                    # Add to current group - same target pitch
                    current_fixed_group.append(segment)
                else:
                    # Process current group if it exists
                    if current_fixed_group:
                        merged_seg = self._merge_fixed_segment_group(current_fixed_group, target_pitches, target_log_freqs)
                        merged_segments.append(merged_seg)
                    
                    # Start new group
                    current_fixed_group = [segment]
                    current_target_pitch_idx = seg_target_idx
            else:
                # Moving segment - process any current fixed group first, then add moving segment as-is
                if current_fixed_group:
                    merged_seg = self._merge_fixed_segment_group(current_fixed_group, target_pitches, target_log_freqs)
                    merged_segments.append(merged_seg)
                    current_fixed_group = []
                    current_target_pitch_idx = None
                
                # Add moving segment as-is (no merging)
                merged_segments.append(segment)
        
        # Don't forget the last fixed group
        if current_fixed_group:
            merged_seg = self._merge_fixed_segment_group(current_fixed_group, target_pitches, target_log_freqs)
            merged_segments.append(merged_seg)
        
        return merged_segments
    
    def _detect_vibrato_segments(self, stage1_segments: List[Dict[str, Any]], 
                                raga_object: IDTAPRaga) -> List[Dict[str, Any]]:
        """
        Detect vibrato segments from Stage 1 segments.
        
        Vibrato criteria:
        1. Sequence of ≥6 consecutive segments
        2. All segments have duration < 0.15s  
        3. Contains mix of 'fixed' and 'moving' segment types
        4. All segments oscillate between at most 2 adjacent pitches
        
        Args:
            stage1_segments: List of segments from Stage 1
            raga_object: IDTAP Raga object for pitch analysis
            
        Returns:
            List of segments with vibrato sequences replaced by single vibrato segments
        """
        if len(stage1_segments) < 6:
            return stage1_segments
        
        # Get target pitches for pitch analysis
        pitch_data_all = np.concatenate([seg['data'] for seg in stage1_segments if 'data' in seg])
        if len(pitch_data_all) == 0:
            return stage1_segments
            
        min_freq = np.nanmin(pitch_data_all) if not np.isnan(np.nanmin(pitch_data_all)) else 75
        max_freq = np.nanmax(pitch_data_all) if not np.isnan(np.nanmax(pitch_data_all)) else 800
        target_pitches = raga_object.get_pitches(low=max(min_freq*0.8, 75), 
                                                high=min(max_freq*1.2, 2400))
        target_log_freqs = [p.non_offset_log_freq for p in target_pitches]
        
        result_segments = []
        i = 0
        
        while i < len(stage1_segments):
            # Look for potential vibrato sequence starting at position i
            vibrato_result = self._find_vibrato_sequence(
                stage1_segments, i, target_pitches, target_log_freqs
            )
            
            if vibrato_result is not None:
                # Found vibrato sequence
                vibrato_segment, vibrato_start_offset, vibrato_end_idx = vibrato_result
                
                # Add any segments before the vibrato sequence
                for j in range(i, i + vibrato_start_offset):
                    result_segments.append(stage1_segments[j])
                
                # Add the vibrato segment
                result_segments.append(vibrato_segment)
                i = vibrato_end_idx  # Skip to after the vibrato sequence
            else:
                # No vibrato, add current segment as-is
                result_segments.append(stage1_segments[i])
                i += 1
        
        return result_segments
    
    def _find_vibrato_sequence(self, segments: List[Dict[str, Any]], start_idx: int,
                              target_pitches: List[Pitch], target_log_freqs: List[float]) -> Optional[Tuple[Dict[str, Any], int, int]]:
        """
        Look for vibrato sequence starting at start_idx.
        
        Finds the longest valid subsequence within consecutive short segments.
        
        Returns:
            Tuple of (vibrato_segment, start_offset, end_idx) if found, None otherwise
            start_offset: number of segments to skip before vibrato starts
            end_idx: absolute index after vibrato sequence ends
        """
        min_segments = 3
        max_duration = 0.20  # seconds
        max_vibrato_range = 0.125  # log frequency units (~1.5 semitones)
        
        # Collect all consecutive short segments from start_idx
        seq_end = start_idx
        candidate_segments = []
        
        while (seq_end < len(segments) and 
               segments[seq_end]['type'] in ['fixed', 'moving'] and
               segments[seq_end]['duration'] < max_duration):
            candidate_segments.append(segments[seq_end])
            seq_end += 1
        
        # Check minimum length for the entire sequence
        if len(candidate_segments) < min_segments:
            return None
        
        # Find the longest valid subsequence that meets all criteria
        best_vibrato = None
        best_end_idx = start_idx
        
        # Try all possible subsequences of length >= min_segments
        for subseq_start in range(len(candidate_segments) - min_segments + 1):
            for subseq_end in range(subseq_start + min_segments, len(candidate_segments) + 1):
                subseq = candidate_segments[subseq_start:subseq_end]
                
                # Check that segments are pitch-based (not silence)
                segment_types = {seg['type'] for seg in subseq}
                if not (segment_types & {'fixed', 'moving'}):  # Must have at least one pitched segment
                    continue
                
                # Check frequency range constraint
                if not self._is_within_vibrato_range(subseq, max_vibrato_range):
                    continue
                    
                # Check for oscillation pattern
                if not self._has_oscillation_pattern(subseq, target_pitches, target_log_freqs):
                    continue
                
                # This is a valid vibrato subsequence
                # Keep the longest one found so far
                if best_vibrato is None or len(subseq) > len(best_vibrato[3]):
                    vibrato_segment = self._create_vibrato_segment(subseq, target_pitches, target_log_freqs)
                    actual_end_idx = start_idx + subseq_start + len(subseq)
                    best_vibrato = (vibrato_segment, subseq_start, actual_end_idx, subseq)
        
        if best_vibrato is not None:
            return best_vibrato[0], best_vibrato[1], best_vibrato[2]  # Return (vibrato_segment, start_offset, end_idx)
        
        return None
    
    def _get_unique_pitches_from_segments(self, segments: List[Dict[str, Any]], 
                                        target_pitches: List[Pitch], 
                                        target_log_freqs: List[float]) -> List[int]:
        """Get unique target pitch indices from a list of segments."""
        pitch_indices = set()
        
        for seg in segments:
            if seg['type'] == 'fixed' and 'pitch' in seg:
                # Find closest target pitch for this fixed segment
                seg_pitch_log = seg['pitch'].non_offset_log_freq
                closest_idx = np.argmin([abs(seg_pitch_log - tlf) for tlf in target_log_freqs])
                pitch_indices.add(closest_idx)
            elif seg['type'] == 'moving':
                # Use representative frequency to find target pitch
                rep_freq = self._get_segment_representative_log_frequency(seg)
                if rep_freq > 0:
                    closest_idx = np.argmin([abs(rep_freq - tlf) for tlf in target_log_freqs])
                    pitch_indices.add(closest_idx)
        
        return sorted(list(pitch_indices))
    
    def _is_within_vibrato_range(self, segments: List[Dict[str, Any]], max_range: float) -> bool:
        """Check if all segments fall within a small frequency range suitable for vibrato."""
        # Use endpoint frequencies like in oscillation pattern detection
        endpoint_freqs = []
        
        for seg in segments:
            if seg['type'] == 'fixed' and 'pitch' in seg:
                endpoint_freqs.append(seg['pitch'].non_offset_log_freq)
            elif seg['type'] == 'moving':
                # Add start frequency of this segment
                endpoint_freqs.append(seg['start_pitch'].non_offset_log_freq)
        # Add the final endpoint of the last segment
        if segments and segments[-1]['type'] == 'moving':
            endpoint_freqs.append(segments[-1]['end_pitch'].non_offset_log_freq)
        
        if len(endpoint_freqs) < 2:
            return False
        
        freq_range = max(endpoint_freqs) - min(endpoint_freqs)
        return freq_range <= max_range
    
    def _has_oscillation_pattern(self, segments: List[Dict[str, Any]], 
                                target_pitches: List[Pitch], 
                                target_log_freqs: List[float]) -> bool:
        """
        Check if segments show true oscillation pattern:
        - Should alternate between higher and lower frequency ranges
        - At least 2 complete cycles (4 direction changes)
        """
        if len(segments) < 5:
            return False
        
        # Get endpoint frequencies to track actual oscillation pattern
        endpoint_freqs = []
        for seg in segments:
            if seg['type'] == 'fixed' and 'pitch' in seg:
                endpoint_freqs.append(seg['pitch'].non_offset_log_freq)
            elif seg['type'] == 'moving':
                # Add start frequency of this segment
                endpoint_freqs.append(seg['start_pitch'].non_offset_log_freq)
            else:
                return False
        # Add the final endpoint of the last segment
        if segments[-1]['type'] == 'moving':
            endpoint_freqs.append(segments[-1]['end_pitch'].non_offset_log_freq)
        
        if len(endpoint_freqs) < 5:
            return False
        
        # Calculate center frequency
        center_freq = (max(endpoint_freqs) + min(endpoint_freqs)) / 2.0
        
        # Classify each endpoint as above or below center
        positions = ['high' if freq > center_freq else 'low' for freq in endpoint_freqs]
        
        # Count transitions between high and low
        transitions = 0
        for i in range(1, len(positions)):
            if positions[i] != positions[i-1]:
                transitions += 1
        
        # Need at least 4 transitions for 2 complete oscillation cycles
        return transitions >= 4
    
    def _are_pitches_adjacent(self, pitch_indices: List[int], target_pitches: List[Pitch]) -> bool:
        """Check if pitch indices represent adjacent pitches in the raga."""
        if len(pitch_indices) != 2:
            return False
        
        idx1, idx2 = pitch_indices
        # Check if the indices are consecutive (allowing for wrap-around within octave)
        return abs(idx1 - idx2) == 1
    
    def _create_vibrato_segment(self, segments: List[Dict[str, Any]], 
                               target_pitches: List[Pitch], 
                               target_log_freqs: List[float]) -> Dict[str, Any]:
        """
        Create a vibrato segment from a sequence of segments.
        
        Calculates:
        - Average pitch (center of vibrato)  
        - Number of oscillations
        - Maximum offset from center
        """
        # Timing info
        start_time = segments[0]['start_time']
        end_time = segments[-1]['start_time'] + segments[-1]['duration']
        duration = end_time - start_time
        
        # Combine all pitch data
        all_pitch_data = []
        for seg in segments:
            if 'data' in seg:
                all_pitch_data.extend(seg['data'])
        
        # Filter valid frequencies
        valid_freqs = [f for f in all_pitch_data if not np.isnan(f) and f > 0]
        
        if not valid_freqs:
            # Fallback - use first segment's pitch
            center_pitch = segments[0].get('pitch', target_pitches[0])
            oscillations = len(segments) // 2
            max_offset = 0.0
        else:
            # Calculate center frequency (average)
            center_freq = np.mean(valid_freqs)
            center_log_freq = np.log2(center_freq)
            
            # Find closest target pitch for center
            closest_idx = np.argmin([abs(center_log_freq - tlf) for tlf in target_log_freqs])
            center_pitch = target_pitches[closest_idx]
            
            # Count oscillations (rough estimate: transitions between segment types)
            oscillations = 0
            for i in range(1, len(segments)):
                if segments[i]['type'] != segments[i-1]['type']:
                    oscillations += 1
            oscillations = max(oscillations // 2, 1)  # Divide by 2 for full cycles
            
            # Calculate maximum offset from center
            log_freqs = [np.log2(f) for f in valid_freqs]
            max_offset = (max(log_freqs) - min(log_freqs)) / 2.0 if len(log_freqs) > 1 else 0.0
        
        return {
            'type': 'vibrato',
            'start_time': float(start_time),
            'duration': float(duration),
            'center_pitch': center_pitch,
            'oscillations': int(oscillations),
            'max_offset': float(max_offset),
            'in_raga': True,  # Vibrato in raga context
            '_start_idx': segments[0].get('_start_idx', 0),
            '_end_idx': segments[-1].get('_end_idx', 0),
            'data': np.array(all_pitch_data)
        }
    
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
        
        # Check if there exists ANY target pitch such that ALL frequencies are within threshold of that SAME target
        fixed_target_pitch = None
        fixed_target_log_freq = None
        
        for target_idx, target_log_freq in enumerate(target_log_freqs):
            # Check if ALL frequencies in the segment are within threshold of this specific target
            all_within_this_target = all(abs(freq - target_log_freq) <= self.log_freq_threshold 
                                       for freq in valid_log_freqs)
            if all_within_this_target:
                fixed_target_pitch = target_pitches[target_idx]
                fixed_target_log_freq = target_log_freq
                break
        
        if fixed_target_pitch is not None:
            # Fixed segment - all frequencies within threshold distance of the same target pitch
            mean_log_offset = mean_log_freq - fixed_target_log_freq
            return [self._create_fixed_pitch_segment(
                start_idx, end_idx, time_data, pitch_data[start_idx:end_idx],
                fixed_target_pitch, mean_log_offset, True
            )]

        # Majority-based fixed heuristic: if start and end lie on the same target and
        # the majority of frames (> fixed_majority_ratio) are within threshold to that target,
        # classify as fixed. This reduces false "moving" for tiny jitter.
        start_target_idx = int(np.argmin([abs(valid_log_freqs[0] - tlf) for tlf in target_log_freqs]))
        end_target_idx = int(np.argmin([abs(valid_log_freqs[-1] - tlf) for tlf in target_log_freqs]))
        if start_target_idx == end_target_idx:
            candidate_target_log = target_log_freqs[start_target_idx]
            close_frac = np.mean(np.abs(valid_log_freqs - candidate_target_log) <= self.log_freq_threshold)
            if close_frac >= self.fixed_majority_ratio:
                candidate_pitch = target_pitches[start_target_idx]
                mean_log_offset = float(np.mean(valid_log_freqs - candidate_target_log))
                return [self._create_fixed_pitch_segment(
                    start_idx, end_idx, time_data, pitch_data[start_idx:end_idx],
                    candidate_pitch, mean_log_offset, True
                )]

        # Otherwise treat as moving
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

    def _segment_pitched_region_very_fine(self, start_idx: int, end_idx: int, 
                                         pitch_data: np.ndarray, log_pitch_data: np.ndarray,
                                         time_data: np.ndarray, target_pitches: List[Pitch],
                                         target_log_freqs: List[float]) -> List[Dict[str, Any]]:
        """
        Segment a pitched region using very fine local minima and maxima detection.
        Uses gradient-based approach for more sensitivity.
        """
        if end_idx - start_idx < 3:
            return self._create_single_segment_from_region(
                start_idx, end_idx, pitch_data, log_pitch_data, 
                time_data, target_pitches, target_log_freqs
            )
        
        region_log_pitch = log_pitch_data[start_idx:end_idx]
        valid_mask = ~np.isnan(region_log_pitch)
        
        if not np.any(valid_mask):
            return []
        
        # Use gradient-based approach for very fine extrema detection
        extrema_indices = self._find_very_local_extrema(region_log_pitch, valid_mask)
        
        # Convert back to global indices
        extrema_indices = [idx + start_idx for idx in extrema_indices]
        
        # Add start and end points
        segment_boundaries = [start_idx] + extrema_indices + [end_idx]
        segment_boundaries = sorted(list(set(segment_boundaries)))
        
        # Create segments between extrema
        segments = []
        for i in range(len(segment_boundaries) - 1):
            seg_start = segment_boundaries[i]
            seg_end = segment_boundaries[i + 1]
            
            if seg_end > seg_start:
                segment = self._create_single_segment_from_region(
                    seg_start, seg_end, pitch_data, log_pitch_data,
                    time_data, target_pitches, target_log_freqs
                )[0]
                segments.append(segment)
        
        return segments
    
    def _find_very_local_extrema(self, log_pitch_data: np.ndarray, valid_mask: np.ndarray) -> List[int]:
        """Find EVERY local minimum and maximum using scipy.signal.find_peaks."""
        from scipy.signal import find_peaks
        
        extrema = []
        
        if len(log_pitch_data) < 3:
            return extrema
        
        # Create a clean array for peak finding (interpolate over NaN values)
        clean_data = log_pitch_data.copy()
        nan_mask = np.isnan(clean_data)
        
        if np.all(nan_mask):
            return extrema
        
        # Simple interpolation over NaN values for peak detection
        if np.any(nan_mask):
            valid_indices = np.where(~nan_mask)[0]
            if len(valid_indices) < 2:
                return extrema
            clean_data = np.interp(np.arange(len(clean_data)), valid_indices, clean_data[valid_indices])
        
        # Find maxima (peaks) - use larger distance to avoid tiny segments
        maxima_indices, _ = find_peaks(clean_data, distance=20)
        
        # Find minima (peaks in inverted data)
        minima_indices, _ = find_peaks(-clean_data, distance=20)
        
        # Combine and sort all extrema
        all_extrema = np.concatenate([maxima_indices, minima_indices])
        extrema = sorted(all_extrema.tolist())
        
        # Filter out indices that correspond to NaN values in original data
        extrema = [i for i in extrema if not np.isnan(log_pitch_data[i])]
        
        return extrema
    
    def _get_segment_target_pitch_index(self, segment: Dict[str, Any], 
                                       target_log_freqs: List[float]) -> int:
        """Get the index of the target pitch closest to this segment."""
        rep_freq = self._get_segment_representative_log_frequency(segment)
        if rep_freq == 0:
            return -1
        return int(np.argmin([abs(rep_freq - tlf) for tlf in target_log_freqs]))
    
    def _is_segment_within_threshold(self, segment: Dict[str, Any], 
                                   target_pitches: List[Pitch], 
                                   target_log_freqs: List[float]) -> bool:
        """Check if segment is within 1/24th octave threshold of a target pitch."""
        rep_freq = self._get_segment_representative_log_frequency(segment)
        if rep_freq == 0:
            return False
        
        # Find distance to closest target
        min_distance = min([abs(rep_freq - tlf) for tlf in target_log_freqs])
        return min_distance <= self.log_freq_threshold
    
    def _merge_fixed_segment_group(self, segment_group: List[Dict[str, Any]], 
                                  target_pitches: List[Pitch], 
                                  target_log_freqs: List[float]) -> Dict[str, Any]:
        """
        Merge a group of consecutive 'fixed' segments that all have the same target pitch.
        Only called for groups of fixed segments, so the result is always a fixed segment.
        """
        if len(segment_group) == 1:
            # Single segment - just return it as-is
            return segment_group[0].copy()
        
        # Merge multiple fixed segments with the same target pitch
        first_seg = segment_group[0]
        last_seg = segment_group[-1]
        
        # Combine data arrays
        all_data = []
        for seg in segment_group:
            if 'data' in seg:
                all_data.extend(seg['data'])
        
        valid_data = [d for d in all_data if not np.isnan(d) and d > 0]
        
        # Calculate merged segment properties
        merged_segment = {
            'type': 'fixed',
            'start_time': first_seg['start_time'],
            'duration': (last_seg['start_time'] + last_seg['duration']) - first_seg['start_time'],
            'in_raga': first_seg.get('in_raga', True),
            '_start_idx': first_seg['_start_idx'],
            '_end_idx': last_seg['_end_idx'],
            'data': np.array(all_data)
        }
        
        # Determine target pitch and log offset for the merged segment
        if valid_data:
            mean_log_freq = np.mean([np.log2(d) for d in valid_data])
            closest_idx = np.argmin([abs(mean_log_freq - tlf) for tlf in target_log_freqs])
            closest_target = target_pitches[closest_idx]
            closest_log_freq = target_log_freqs[closest_idx]
            
            merged_segment.update({
                'pitch': closest_target,
                'log_offset': mean_log_freq - closest_log_freq
            })
        else:
            # Fallback - use the first segment's pitch info
            merged_segment.update({
                'pitch': first_seg.get('pitch', target_pitches[0]),
                'log_offset': first_seg.get('log_offset', 0.0)
            })
        
        return merged_segment


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
