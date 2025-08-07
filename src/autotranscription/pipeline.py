"""
Auto-transcription pipeline for Indian classical music - Refactored Version.

Main orchestrator using modular components for better organization and maintainability.
"""

import time
import json
import shutil
from pathlib import Path
from typing import Union, Dict, Any, Optional
import numpy as np

def _raise_json_error(o: Any):
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
import numpy as np

try:
    from .audio_separator import AudioSeparator
    from .pitch_extraction import PitchExtractor
    from .pitch_cleaning import PitchCleaner
    from .tonic_estimation import TonicEstimator
    from .visualization import SargamVisualizer
    from .audio_processing import AudioProcessor
    from .raga_estimation import RagaEstimator
    from .pitch_segmentation import PitchSegmenter
except ImportError:
    from audio_separator import AudioSeparator
    from pitch_extraction import PitchExtractor
    from pitch_cleaning import PitchCleaner
    from tonic_estimation import TonicEstimator
    from visualization import SargamVisualizer
    from audio_processing import AudioProcessor
    from raga_estimation import RagaEstimator
    from pitch_segmentation import PitchSegmenter

# Import IDTAP Raga class
try:
    # Add the Python-API path to handle the installed idtap package
    import sys
    sys.path.insert(0, '/Users/jon/documents/2025/idtap_project/Python-API')
    from idtap.classes.raga import Raga as IDTAPRaga
    from idtap.client import SwaraClient
except ImportError:
    print("Warning: idtap not available. Raga object creation will be skipped.")
    IDTAPRaga = None
    SwaraClient = None


class AutoTranscriptionPipeline:
    """
    Complete pipeline for auto-transcription of Indian classical music.
    
    Orchestrates the following stages:
    1. Audio separation (vocals/instrumental)
    2. Tonic estimation
    3. Pitch trace extraction
    3.5. Pitch cleaning (octave jump correction)
    4. Raga estimation using TDMS
    5. Pitch segmentation (fixed/moving/silence)
    6. Visualization generation
    7. Audio output creation
    """
    
    def __init__(self, audio_separator_model: str = "htdemucs", workspace_dir: Union[str, Path] = None, 
                 algorithms: Optional[list] = None):
        """
        Initialize the pipeline with modular components.
        
        Args:
            audio_separator_model: Demucs model to use for audio separation
            workspace_dir: Directory to store all pipeline files (optional)
            algorithms: List of algorithms to run. Default: ['essentia'] 
                       Available: ['essentia', 'swipe', 'pyworld', 'crepe']
        """
        # Initialize workspace
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "pipeline_workspace"
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.audio_dir = self.workspace_dir / "audio"
        self.metadata_dir = self.workspace_dir / "metadata"
        self.separated_dir = self.workspace_dir / "separated"
        self.data_dir = self.workspace_dir / "data"
        self.pitch_traces_dir = self.data_dir / "pitch_traces"
        self.segmentation_dir = self.data_dir / "segmentation"
        self.visualizations_dir = self.workspace_dir / "visualizations"
        self.audio_outputs_dir = self.workspace_dir / "audio_outputs"
        
        for dir_path in [self.audio_dir, self.metadata_dir, self.separated_dir, 
                        self.data_dir, self.pitch_traces_dir, self.segmentation_dir,
                        self.visualizations_dir, self.audio_outputs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize modular components
        self.audio_separator = AudioSeparator(model_name=audio_separator_model)
        self.pitch_extractor = PitchExtractor(algorithms=algorithms)
        self.pitch_cleaner = PitchCleaner()
        self.tonic_estimator = TonicEstimator()
        self.raga_estimator = RagaEstimator(models_dir=self.workspace_dir / "raga_models")
        self.visualizer = SargamVisualizer(self.visualizations_dir)
        self.audio_processor = AudioProcessor()
        
        # Store current Raga object (not persisted in JSON)
        self.current_raga_object = None
    
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
            json.dump(
                metadata,
                f,
                indent=2,
                default=lambda o: o.item() if isinstance(o, np.generic) else (_raise_json_error(o))
            )
    
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
        
        # Stage 1: Audio Separation
        if "audio_separation" not in metadata["stages_completed"]:
            separation_results = self._run_audio_separation(audio_id, metadata)
            metadata["results"]["audio_separation"] = separation_results
            metadata["stages_completed"].append("audio_separation")
            self.save_metadata(audio_id, metadata)
        else:
            separation_results = metadata["results"]["audio_separation"]
        
        # Stage 2: Tonic Estimation
        if "tonic_estimation" not in metadata["stages_completed"]:
            tonic_results = self._run_tonic_estimation(audio_id, metadata)
            metadata["results"]["tonic_estimation"] = tonic_results
            metadata["stages_completed"].append("tonic_estimation")
            self.save_metadata(audio_id, metadata)
        else:
            tonic_results = metadata["results"]["tonic_estimation"]
        
        # Stage 3: Pitch Trace Extraction
        if "pitch_trace" not in metadata["stages_completed"]:
            pitch_results = self._run_pitch_trace_extraction(audio_id, metadata)
            metadata["results"]["pitch_trace"] = pitch_results
            metadata["stages_completed"].append("pitch_trace")
            self.save_metadata(audio_id, metadata)
        else:
            pitch_results = metadata["results"]["pitch_trace"]
        
        # Stage 3.5: Pitch Cleaning (Octave Jump Correction)
        if "pitch_cleaning" not in metadata["stages_completed"]:
            cleaning_results = self._run_pitch_cleaning(audio_id, metadata)
            metadata["results"]["pitch_cleaning"] = cleaning_results
            metadata["stages_completed"].append("pitch_cleaning")
            self.save_metadata(audio_id, metadata)
        else:
            cleaning_results = metadata["results"]["pitch_cleaning"]
        
        # Stage 4: Raga Estimation
        if "raga_estimation" not in metadata["stages_completed"]:
            raga_results = self._run_raga_estimation(audio_id, metadata)
            metadata["results"]["raga_estimation"] = raga_results
            metadata["stages_completed"].append("raga_estimation")
            self.save_metadata(audio_id, metadata)
        else:
            raga_results = metadata["results"]["raga_estimation"]
            # Recreate Raga object if it was previously created but not in memory
            if (raga_results.get('raga_object_created', False) and 
                IDTAPRaga is not None and 
                self.current_raga_object is None):
                try:
                    predicted_raga_name = raga_results['predicted_raga']
                    tonic_freq = metadata["results"]["tonic_estimation"]["original"]["tonic_frequency"]
                    
                    # Try to get raga from API first
                    api_raga = self._get_raga_from_api(predicted_raga_name, tonic_freq)
                    if api_raga:
                        self.current_raga_object = api_raga
                        print(f"🌐 Using API definition for {predicted_raga_name} (tonic: {tonic_freq:.1f} Hz)")
                    else:
                        # Fall back to basic raga object
                        self.current_raga_object = IDTAPRaga({
                            'name': predicted_raga_name,
                            'fundamental': tonic_freq
                        })
                        print(f"🔄 Recreated IDTAP Raga object: {predicted_raga_name} (tonic: {tonic_freq:.1f} Hz)")
                    
                    # Print out the raga pitch rules for verification
                    print(f"\n📜 {predicted_raga_name} Raga Pitch Rules:")
                    print("-" * 60)
                    try:
                        # Get pitches in a reasonable range around the tonic
                        min_freq = max(tonic_freq * 0.5, 75)  # Half octave below
                        max_freq = min(tonic_freq * 4, 1200)   # Two octaves above
                        raga_pitches = self.current_raga_object.get_pitches(low=min_freq, high=max_freq)
                        
                        print(f"Tonic frequency: {tonic_freq:.1f} Hz")
                        print(f"Pitch range: {min_freq:.1f} Hz to {max_freq:.1f} Hz")
                        print(f"Total pitches: {len(raga_pitches)}")
                        print()
                        
                        # Calculate tonic log frequency for cent calculations
                        import math
                        tonic_log_freq = math.log2(tonic_freq)
                        
                        # Group pitches by octave for better readability
                        pitches_by_oct = {}
                        for pitch in raga_pitches:
                            oct = pitch.oct
                            if oct not in pitches_by_oct:
                                pitches_by_oct[oct] = []
                            pitches_by_oct[oct].append(pitch)
                        
                        # Print pitches by octave
                        for oct in sorted(pitches_by_oct.keys()):
                            print(f"  Octave {oct}:")
                            octave_pitches = sorted(pitches_by_oct[oct], key=lambda p: p.frequency)
                            for pitch in octave_pitches:
                                sargam = pitch.octaved_sargam_letter
                                freq = pitch.frequency
                                # Calculate cents from tonic
                                cents_from_tonic = 1200 * (pitch.log_freq - tonic_log_freq)
                                print(f"    {sargam:>4} = {freq:>7.1f} Hz (tonic + {cents_from_tonic:>6.1f} cents)")
                        print("-" * 60)
                        
                        # Also show the swara sequence
                        print("Swara sequence (ascending order by frequency):")
                        all_pitches = sorted(raga_pitches, key=lambda p: p.frequency)
                        swara_seq = [p.sargam_letter for p in all_pitches]
                        unique_swaras = []
                        for swara in swara_seq:
                            if not unique_swaras or unique_swaras[-1] != swara:
                                unique_swaras.append(swara)
                        print("  " + " - ".join(unique_swaras))
                        print("-" * 60)
                        
                    except Exception as e:
                        print(f"⚠️  Could not retrieve pitch rules: {e}")
                        
                except Exception as e:
                    print(f"⚠️  Failed to recreate IDTAP Raga object: {e}")
        
        # Stage 5: Pitch Segmentation - Force regeneration with new min_duration
        print(f"🔄 Running pitch segmentation with 0.25s minimum duration...")
        
        # Always remove from completed stages to force regeneration with new parameters
        if "pitch_segmentation" in metadata["stages_completed"]:
            metadata["stages_completed"].remove("pitch_segmentation")
        
        # Also remove existing results to ensure clean regeneration
        if "pitch_segmentation" in metadata.get("results", {}):
            del metadata["results"]["pitch_segmentation"]
            
        if "pitch_segmentation" not in metadata["stages_completed"]:
            segmentation_results = self._run_pitch_segmentation(audio_id, metadata)
            metadata["results"]["pitch_segmentation"] = segmentation_results
            metadata["stages_completed"].append("pitch_segmentation")
            self.save_metadata(audio_id, metadata)
        else:
            segmentation_results = metadata["results"]["pitch_segmentation"]
        
        pipeline_time = time.time() - pipeline_start
        metadata["last_run_time"] = pipeline_time
        self.save_metadata(audio_id, metadata)
        
        return metadata
    
    def _run_audio_separation(self, audio_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run audio separation stage using AudioSeparator."""
        workspace_audio_path = Path(metadata["workspace_file"])
        separation_output_dir = self.separated_dir / str(audio_id)
        
        results = self.audio_separator.separate_vocals(workspace_audio_path, separation_output_dir)
        
        # Update paths to be relative to workspace
        results["vocals"] = str(results["vocals"])
        results["instrumental"] = str(results["instrumental"])
        results["output_dir"] = str(results["output_dir"])
        
        return results
    
    def _run_tonic_estimation(self, _audio_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run tonic estimation stage using TonicEstimator."""
        original_path = Path(metadata["workspace_file"])
        separation_results = metadata["results"]["audio_separation"]
        instrumental_path = Path(separation_results["instrumental"])
        
        return self.tonic_estimator.estimate_tonic_from_files(original_path, instrumental_path)
    
    def _run_pitch_trace_extraction(self, audio_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run pitch trace extraction stage using PitchExtractor."""
        separation_results = metadata["results"]["audio_separation"]
        vocals_path = Path(separation_results["vocals"])
        
        # Extract pitch traces using all selected algorithms
        pitch_results = self.pitch_extractor.extract_all_algorithms(
            vocals_path, audio_id, self.pitch_traces_dir
        )
        
        # Create visualizations using SargamVisualizer
        confidence_thresholds = self.pitch_extractor.get_confidence_thresholds()
        self.visualizer.create_pitch_visualization(
            audio_id, metadata, pitch_results, confidence_thresholds
        )
        
        # Create audio outputs using AudioProcessor
        self.audio_processor.create_pitch_audio_outputs(
            audio_id, vocals_path, pitch_results, self.audio_outputs_dir, confidence_thresholds
        )
        
        return pitch_results
    
    def _run_pitch_cleaning(self, audio_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run pitch cleaning stage to remove octave jumps and artifacts."""
        pitch_results = metadata["results"]["pitch_trace"]
        
        cleaning_results = {}
        
        # Clean pitch data for each algorithm that was used
        for algorithm_key, algorithm_data in pitch_results.items():
            if not algorithm_key.startswith('vocals_'):
                continue
                
            # Load the original pitch data
            pitch_file = Path(algorithm_data["pitch_data_file"])
            confidence_file = Path(algorithm_data["confidence_data_file"])
            times_file = Path(algorithm_data["times_data_file"])
            
            if not all(f.exists() for f in [pitch_file, confidence_file, times_file]):
                print(f"⚠️  Skipping cleaning for {algorithm_key}: missing data files")
                continue
            
            print(f"🧹 Cleaning pitch data for {algorithm_key}...")
            
            # Load pitch data
            pitch_data = np.load(pitch_file)
            confidence_data = np.load(confidence_file)
            times_data = np.load(times_file)
            
            # Clean the pitch data
            cleaned_pitch, cleaning_stats = self.pitch_cleaner.clean_pitch_trace(
                pitch_data, confidence_data, times_data
            )
            
            # Save cleaned data and get file information
            algorithm_name = algorithm_key.replace('vocals_', '')
            clean_results = self.pitch_cleaner.save_cleaning_results(
                audio_id, algorithm_name, cleaned_pitch, cleaning_stats, self.pitch_traces_dir
            )
            
            # Store results
            cleaning_results[algorithm_key] = {
                **clean_results,
                'original_pitch_file': str(pitch_file),
                'original_octave_jumps': cleaning_stats['octave_jumps_detected'],
                'octave_jumps_corrected': cleaning_stats['octave_jumps_corrected'],
                'correction_rate': (cleaning_stats['octave_jumps_corrected'] / 
                                  max(cleaning_stats['octave_jumps_detected'], 1)) * 100
            }
            
            print(f"  ✅ Detected {cleaning_stats['octave_jumps_detected']} octave jumps")
            print(f"  ✅ Corrected {cleaning_stats['octave_jumps_corrected']} octave jumps")
            if cleaning_stats['corrections_applied']:
                total_corrected_duration = sum(
                    c['duration_frames'] for c in cleaning_stats['corrections_applied']
                )
                print(f"  ✅ Total corrected duration: {total_corrected_duration} frames")
        
        print(f"🧹 Pitch cleaning completed for {len(cleaning_results)} algorithms")
        
        return cleaning_results
    
    def _run_raga_estimation(self, audio_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run raga estimation stage using RagaEstimator with TDMS."""
        tonic_freq = metadata["results"]["tonic_estimation"]["original"]["tonic_frequency"]
        pitch_results = metadata["results"]["pitch_trace"]
        
        # Try to estimate raga from the best available pitch extraction algorithm
        # Prefer Essentia, but use any available algorithm
        algorithm_priority = ['vocals_essentia', 'vocals_crepe', 'vocals_pyworld', 'vocals_swipe']
        
        raga_results = {}
        for algorithm_key in algorithm_priority:
            if algorithm_key in pitch_results:
                try:
                    # Get pitch data files
                    pitch_file = pitch_results[algorithm_key]['pitch_data_file']
                    confidence_file = pitch_results[algorithm_key]['confidence_data_file']
                    times_file = pitch_results[algorithm_key]['times_data_file']
                    
                    # Get algorithm-specific confidence threshold
                    confidence_thresholds = self.pitch_extractor.get_confidence_thresholds()
                    confidence_threshold = confidence_thresholds.get(algorithm_key, 0.1)
                    
                    # Estimate raga using TDMS
                    raga_result = self.raga_estimator.estimate_raga_from_pitch_data(
                        pitch_file, confidence_file, times_file, 
                        tonic_freq, confidence_threshold
                    )
                    
                    # If we got a valid result, store it and break
                    if raga_result and 'predicted_raga' in raga_result:
                        raga_results = raga_result  # Use the result directly
                        break
                        
                except Exception as e:
                    # Continue to next algorithm if this one fails
                    continue
        
        # If no successful results, return error
        if not raga_results or 'predicted_raga' not in raga_results:
            raga_results = {
                "error": "Raga estimation failed - insufficient pitch data or processing error"
            }
        else:
            # Create IDTAP Raga object - try to get from API first, fallback to local
            if IDTAPRaga is not None:
                try:
                    predicted_raga_name = raga_results['predicted_raga']
                    
                    # Try to get raga definition from API
                    raga_obj = self._get_raga_from_api(predicted_raga_name, tonic_freq)
                    
                    if raga_obj is None:
                        # Fallback to local raga definition
                        print(f"⚠️  Could not fetch {predicted_raga_name} from API, using local definition")
                        raga_obj = IDTAPRaga({
                            'name': predicted_raga_name,
                            'fundamental': tonic_freq
                        })
                    
                    # Store the Raga object in the pipeline instance (not in JSON metadata)
                    self.current_raga_object = raga_obj
                    raga_results['raga_object_created'] = True
                    print(f"✅ Created IDTAP Raga object: {predicted_raga_name} (tonic: {tonic_freq:.1f} Hz)")
                    print(f"   Available sargam letters: {raga_obj.sargam_letters}")
                    print(f"   Fundamental frequency: {raga_obj.fundamental:.1f} Hz")
                except Exception as e:
                    print(f"⚠️  Failed to create IDTAP Raga object: {e}")
                    raga_results['raga_object_created'] = False
                    self.current_raga_object = None
            else:
                raga_results['raga_object_created'] = False
                self.current_raga_object = None
        
        return raga_results
    
    def _get_raga_from_api(self, raga_name: str, tonic_freq: float) -> Optional[IDTAPRaga]:
        """Try to get raga definition from the Swara Studio API."""
        if SwaraClient is None:
            return None
        
        try:
            # Initialize client (will attempt login if needed)
            client = SwaraClient(auto_login=True)
            
            # Get available ragas
            available_ragas = client.get_available_ragas()
            
            if raga_name in available_ragas:
                print(f"🌐 Found {raga_name} in API, fetching definition...")
                
                # Get raga rules from API
                raga_rules = client.get_raga_rules(raga_name)
                
                if raga_rules:
                    print(f"🌐 Raw API rules for {raga_name}:")
                    print(f"  {raga_rules}")
                    
                    # Extract the actual rules from the nested structure
                    actual_rules = raga_rules.get('rules', raga_rules)
                    print(f"🌐 Extracted rules for IDTAP:")
                    print(f"  {actual_rules}")
                    
                    # Create IDTAP Raga object with the API rules and tonic frequency
                    raga_data = {
                        'name': raga_name,
                        'fundamental': tonic_freq,
                        'rule_set': actual_rules
                    }
                    api_raga = IDTAPRaga(raga_data)
                    print(f"🌐 Successfully created {raga_name} from API definition")
                    return api_raga
                else:
                    print(f"🌐 No rules found for {raga_name} in API")
                    return None
            else:
                print(f"🌐 {raga_name} not found in API ({len(available_ragas)} ragas available)")
                return None
                
        except Exception as e:
            print(f"🌐 Failed to connect to API for raga definition: {e}")
            return None
    
    def _run_pitch_segmentation(self, audio_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run pitch segmentation stage using PitchSegmenter."""
        # Ensure we have a Raga object
        if self.current_raga_object is None:
            return {
                "error": "Cannot perform pitch segmentation - no IDTAP Raga object available"
            }
        
        try:
            import numpy as np
            
            # Initialize pitch segmenter
            segmenter = PitchSegmenter(min_duration=0.05)  # 50ms minimum
            
            # Approach 1: Original approach with post-processing
            print("🔄 Running original segmentation approach...")
            segments_original = segmenter.segment_from_pipeline_data(
                audio_id, self.workspace_dir, self.current_raga_object
            )
            
            # Save and visualize original approach
            self._save_segmentation_data(audio_id, segments_original, suffix="_original")
            self._create_segmentation_visualization(audio_id, segments_original, suffix="_original")
            
            # Approach 2: Two-stage Minima/Maxima approach
            print("🔄 Running two-stage minima/maxima segmentation approach...")
            
            # Load pitch and confidence data for minmax approach
            traces_dir = self.workspace_dir / "data" / "pitch_traces"
            pitch_file = traces_dir / f"{audio_id}_vocals_essentia.npy"
            conf_file = traces_dir / f"{audio_id}_vocals_essentia_confidence.npy"
            time_file = traces_dir / f"{audio_id}_vocals_essentia_times.npy"
            
            pitch_data = np.load(pitch_file)
            time_data = np.load(time_file)
            
            # Apply the SAME confidence-based silence filter used in visualization
            if conf_file.exists():
                confidence = np.load(conf_file)
                conf_thresholds = self.pitch_extractor.get_confidence_thresholds()
                ess_thr = float(conf_thresholds.get('vocals_essentia', 0.05))
                low_conf_mask = confidence <= ess_thr
                # Set low-confidence frames to 0.0 so segmentation treats them as silence
                pitch_data = pitch_data.copy()
                pitch_data[low_conf_mask] = 0.0
            
            # Get both stages from the new method
            minmax_results = segmenter.segment_pitch_trace_minmax(
                pitch_data, time_data, self.current_raga_object
            )
            
            # Extract both stages
            stage1_segments = minmax_results['stage1_segments']
            stage2_segments = minmax_results['stage2_segments']
            
            print(f"   Stage 1 (Fine segmentation): {len(stage1_segments)} segments")
            print(f"   Stage 2 (Merged segments): {len(stage2_segments)} segments")
            
            # Save and visualize Stage 1 (fine segmentation)
            self._save_segmentation_data(audio_id, stage1_segments, suffix="_stage1")
            self._create_segmentation_visualization(audio_id, stage1_segments, suffix="_stage1")
            
            # Save and visualize Stage 2 (merged segmentation)  
            self._save_segmentation_data(audio_id, stage2_segments, suffix="_stage2")
            self._create_segmentation_visualization(audio_id, stage2_segments, suffix="_stage2")
            
            # Use Stage 2 (merged) segments for final results
            segments = stage2_segments
            
            # Convert segments to JSON-serializable format
            segments_data = []
            for segment in segments:
                segment_dict = {
                    'type': segment['type'],
                    'start_time': float(segment['start_time']),
                    'duration': float(segment['duration']),
                    'in_raga': bool(segment.get('in_raga', True))
                }
                
                if segment['type'] == 'fixed':
                    segment_dict['pitch'] = segment['pitch'].to_json()
                    segment_dict['log_offset'] = float(segment['log_offset'])
                elif segment['type'] == 'moving':
                    segment_dict['start_pitch'] = segment['start_pitch'].to_json()
                    segment_dict['end_pitch'] = segment['end_pitch'].to_json()
                
                # Note: segment data arrays are not serialized to JSON for space reasons
                # They can be reconstructed from the original pitch trace if needed
                segments_data.append(segment_dict)
            
            results = {
                'segments': segments_data,
                'total_segments': len(segments_data),
                'segment_types': {
                    'fixed': sum(1 for s in segments_data if s['type'] == 'fixed'),
                    'moving': sum(1 for s in segments_data if s['type'] == 'moving'),
                    'silence': sum(1 for s in segments_data if s['type'] == 'silence')
                },
                'in_raga_segments': sum(1 for s in segments_data if s.get('in_raga', True)),
                'out_of_raga_segments': sum(1 for s in segments_data if not s.get('in_raga', True)),
                'total_duration': sum(s['duration'] for s in segments_data)
            }
            
            print(f"✅ Pitch segmentation complete: {results['total_segments']} segments")
            print(f"   Fixed: {results['segment_types']['fixed']}, "
                  f"Moving: {results['segment_types']['moving']}, "
                  f"Silence: {results['segment_types']['silence']}")
            print(f"   In-raga: {results['in_raga_segments']}, "
                  f"Out-of-raga: {results['out_of_raga_segments']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Pitch segmentation failed: {str(e)}"
            print(f"⚠️  {error_msg}")
            return {"error": error_msg}
    
    def _save_segmentation_data(self, audio_id: int, segments: list, suffix: str = ""):
        """Save detailed segmentation data to files for inspection."""
        import json
        import numpy as np
        
        # Create detailed segments file with full data
        segments_file = self.segmentation_dir / f"{audio_id}_segments{suffix}.json"
        detailed_segments = []
        
        for i, segment in enumerate(segments):
            seg_dict = {
                'segment_id': i,
                'type': segment['type'],
                'start_time': float(segment['start_time']),
                'duration': float(segment['duration']),
                'in_raga': bool(segment.get('in_raga', True)),
                'start_idx': int(segment['_start_idx']),
                'end_idx': int(segment['_end_idx'])
            }
            
            # Add pitch information
            if segment['type'] == 'fixed':
                pitch_data = {
                    'swara': segment['pitch'].swara,
                    'oct': segment['pitch'].oct,
                    'raised': segment['pitch'].raised,
                    'sargam_letter': segment['pitch'].sargam_letter,
                    'octaved_sargam_letter': segment['pitch'].octaved_sargam_letter,
                    'frequency': float(segment['pitch'].frequency),
                    'log_freq': float(segment['pitch'].log_freq),
                    'log_offset': float(segment['log_offset'])
                }
                seg_dict['pitch'] = pitch_data
            elif segment['type'] == 'moving':
                seg_dict['start_pitch'] = {
                    'swara': segment['start_pitch'].swara,
                    'oct': segment['start_pitch'].oct,
                    'raised': segment['start_pitch'].raised,
                    'sargam_letter': segment['start_pitch'].sargam_letter,
                    'octaved_sargam_letter': segment['start_pitch'].octaved_sargam_letter,
                    'frequency': float(segment['start_pitch'].frequency),
                    'log_freq': float(segment['start_pitch'].log_freq)
                }
                seg_dict['end_pitch'] = {
                    'swara': segment['end_pitch'].swara,
                    'oct': segment['end_pitch'].oct,
                    'raised': segment['end_pitch'].raised,
                    'sargam_letter': segment['end_pitch'].sargam_letter,
                    'octaved_sargam_letter': segment['end_pitch'].octaved_sargam_letter,
                    'frequency': float(segment['end_pitch'].frequency),
                    'log_freq': float(segment['end_pitch'].log_freq)
                }
            
            detailed_segments.append(seg_dict)
            
            # Save individual segment pitch data arrays
            if 'data' in segment and segment['data'].size > 0:
                segment_data_file = self.segmentation_dir / f"{audio_id}_segment_{i:03d}_data{suffix}.npy"
                np.save(segment_data_file, segment['data'])
        
        # Save the segments JSON file
        with open(segments_file, 'w') as f:
            json.dump(detailed_segments, f, indent=2)
        
        # Create a summary file
        summary_file = self.segmentation_dir / f"{audio_id}_summary{suffix}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Pitch Segmentation Summary for Audio ID: {audio_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total segments: {len(segments)}\n")
            
            types_count = {'fixed': 0, 'moving': 0, 'silence': 0}
            in_raga_count = 0
            out_of_raga_count = 0
            total_duration = 0
            
            for segment in segments:
                types_count[segment['type']] += 1
                if segment.get('in_raga', True):
                    in_raga_count += 1
                else:
                    out_of_raga_count += 1
                total_duration += segment['duration']
            
            f.write(f"Fixed pitch segments: {types_count['fixed']}\n")
            f.write(f"Moving pitch segments: {types_count['moving']}\n")
            f.write(f"Silence segments: {types_count['silence']}\n")
            f.write(f"In-raga segments: {in_raga_count}\n")
            f.write(f"Out-of-raga segments: {out_of_raga_count}\n")
            f.write(f"Total duration: {total_duration:.2f}s\n\n")
            
            f.write("Segment Details:\n")
            f.write("-" * 30 + "\n")
            
            for i, segment in enumerate(segments):
                f.write(f"Segment {i:03d}: {segment['type'].upper()}\n")
                f.write(f"  Time: {segment['start_time']:.3f}s - {segment['start_time'] + segment['duration']:.3f}s ({segment['duration']:.3f}s)\n")
                f.write(f"  In-raga: {segment.get('in_raga', True)}\n")
                
                if segment['type'] == 'fixed':
                    pitch = segment['pitch']
                    f.write(f"  Pitch: {pitch.octaved_sargam_letter} ({pitch.frequency:.1f} Hz)\n")
                    f.write(f"  Log offset: {segment['log_offset']:.4f}\n")
                elif segment['type'] == 'moving':
                    start_pitch = segment['start_pitch']
                    end_pitch = segment['end_pitch']
                    f.write(f"  From: {start_pitch.octaved_sargam_letter} ({start_pitch.frequency:.1f} Hz)\n")
                    f.write(f"  To: {end_pitch.octaved_sargam_letter} ({end_pitch.frequency:.1f} Hz)\n")
                
                f.write("\n")
        
        print(f"💾 Segmentation data saved to: {self.segmentation_dir}")
        print(f"   - {segments_file.name}: Detailed JSON data")
        print(f"   - {summary_file.name}: Human-readable summary")
        print(f"   - Individual segment data arrays: {audio_id}_segment_*.npy")
    
    def _create_segmentation_visualization(self, audio_id: int, segments: list, suffix: str = ""):
        """Create a detailed visualization of pitch segmentation."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        
        try:
            # Load pitch and time data
            pitch_file = self.pitch_traces_dir / f"{audio_id}_vocals_essentia.npy"
            confidence_file = self.pitch_traces_dir / f"{audio_id}_vocals_essentia_confidence.npy"
            times_file = self.pitch_traces_dir / f"{audio_id}_vocals_essentia_times.npy"
            
            pitch_data = np.load(pitch_file)
            confidence_data = np.load(confidence_file)
            
            if times_file.exists():
                time_data = np.load(times_file)
            else:
                # Create time array if not available
                hop_length = 512
                sample_rate = 44100
                time_data = np.arange(len(pitch_data)) * hop_length / sample_rate
            
            # Apply confidence filtering
            confidence_threshold = 0.05
            filtered_pitch = pitch_data.copy()
            low_conf_mask = confidence_data <= confidence_threshold
            filtered_pitch[low_conf_mask] = np.nan
            
            # Convert to log frequency for better visualization
            log_pitch = np.full_like(filtered_pitch, np.nan)
            valid_mask = ~np.isnan(filtered_pitch) & (filtered_pitch > 0)
            log_pitch[valid_mask] = np.log2(filtered_pitch[valid_mask])
            
            # Create figure with extremely wide aspect ratio for maximum detail
            total_duration = time_data[-1]
            fig_width = max(120, total_duration * 2.0)  # Extremely wide: at least 120 inches, 2 inches per second
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 12), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot 1: Pitch trace with segment boundaries
            ax1.plot(time_data, log_pitch, 'b-', linewidth=0.8, alpha=0.7, label='Pitch trace')
            
            # Add raga target pitch lines - make them prominent and comprehensive
            if self.current_raga_object:
                # Get raga pitches across a wider range to show full context
                target_pitches = self.current_raga_object.get_pitches(
                    low=max(np.nanmin(filtered_pitch[valid_mask])*0.7, 50), 
                    high=min(np.nanmax(filtered_pitch[valid_mask])*1.3, 1000)
                )
                target_log_freqs = [p.non_offset_log_freq for p in target_pitches]
                
                print(f"Adding {len(target_pitches)} raga pitch reference lines")
                print("Visualization raga pitches:")
                for pitch in target_pitches:
                    print(f"  {pitch.octaved_sargam_letter} = {pitch.frequency:.1f} Hz (oct: {pitch.oct})")
                
                for i, (pitch, log_freq) in enumerate(zip(target_pitches, target_log_freqs)):
                    # Make raga pitch lines more visible with different colors by octave
                    if pitch.oct <= 0:
                        line_color = 'red'
                        line_alpha = 0.8
                    elif pitch.oct == 1:
                        line_color = 'blue' 
                        line_alpha = 0.9
                    else:
                        line_color = 'green'
                        line_alpha = 0.8
                    
                    # Draw horizontal reference line
                    ax1.axhline(y=log_freq, color=line_color, linestyle='--', 
                               alpha=line_alpha, linewidth=1.0, zorder=1)
                    
                    # Add sargam labels only on the right side to avoid clutter
                    label = f"{pitch.octaved_sargam_letter} ({pitch.frequency:.0f}Hz)"
                    
                    # Right side labels only - no left side repetition
                    ax1.text(total_duration * 1.01, log_freq, label,
                            va='center', ha='left', fontsize=10, color=line_color,
                            fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', alpha=0.8))
                
                # Set custom y-axis labels with sargam names instead of log frequencies
                y_ticks = target_log_freqs
                y_labels = [pitch.octaved_sargam_letter for pitch in target_pitches]
                ax1.set_yticks(y_ticks)
                ax1.set_yticklabels(y_labels)
            
            # Color-code segments and add boundaries
            colors = {'fixed': 'red', 'moving': 'orange', 'silence': 'gray'}
            segment_heights = []
            
            for i, segment in enumerate(segments):
                start_time = segment['start_time']
                end_time = start_time + segment['duration']
                segment_type = segment['type']
                in_raga = segment.get('in_raga', True)
                
                # Add vertical boundary lines
                ax1.axvline(x=start_time, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
                if i == len(segments) - 1:  # Add end line for last segment
                    ax1.axvline(x=end_time, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
                
                # Add segment background shading
                y_min, y_max = ax1.get_ylim()
                alpha_val = 0.3 if in_raga else 0.15
                ax1.axvspan(start_time, end_time, 
                           color=colors[segment_type], alpha=alpha_val, 
                           label=f'{segment_type.title()}' if i == 0 or segment_type != segments[i-1]['type'] else "")
                
                # Add segment number at top
                mid_time = (start_time + end_time) / 2
                ax1.text(mid_time, y_max * 0.98, f'{i}', 
                        ha='center', va='top', fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[segment_type], alpha=0.7))
                
                # Store segment info for bottom plot
                segment_heights.append((start_time, end_time, segment_type, in_raga))
            
            # Customize pitch plot
            ax1.set_ylabel('Sargam', fontsize=12)
            ax1.set_title(f'Pitch Segmentation - Audio ID {audio_id} ({self.current_raga_object.name if self.current_raga_object else "Unknown"} Raga)', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Remove all legends to clean up the visualization - colors are self-explanatory
            
            # Plot 2: Segment type timeline
            for start_time, end_time, segment_type, in_raga in segment_heights:
                y_pos = 0.5
                height = 0.8
                
                # Color and pattern for segment type and raga status
                color = colors[segment_type]
                alpha_val = 0.8 if in_raga else 0.4
                hatch = '' if in_raga else '///'
                
                rect = patches.Rectangle((start_time, y_pos - height/2), 
                                       end_time - start_time, height,
                                       facecolor=color, alpha=alpha_val, 
                                       hatch=hatch, edgecolor='black', linewidth=0.5)
                ax2.add_patch(rect)
                
                # Add segment type label if segment is wide enough
                if (end_time - start_time) > total_duration * 0.03:  # Only if >3% of total duration
                    mid_time = (start_time + end_time) / 2
                    ax2.text(mid_time, y_pos, segment_type[0].upper(), 
                            ha='center', va='center', fontsize=10, fontweight='bold',
                            color='white' if in_raga else 'black')
            
            # Customize timeline plot
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(0, 1)
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Segments', fontsize=12)
            ax2.set_yticks([])
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Remove the legend to save vertical space - colors are self-explanatory
            # Red = Fixed pitch, Orange = Moving pitch, Gray = Silence
            # Solid = In-raga, Hatched = Out-of-raga
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.visualizations_dir / f"{audio_id}_pitch_segmentation{suffix}.png"
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Segmentation visualization saved: {viz_file.name}")
            
        except Exception as e:
            print(f"⚠️  Failed to create segmentation visualization: {e}")
    
    def get_current_raga_object(self):
        """
        Get the current IDTAP Raga object if available.
        
        Returns:
            IDTAP Raga object or None if not available
        """
        return self.current_raga_object
    
    def list_audio_files(self) -> Dict[int, Dict[str, Any]]:
        """
        List all audio files in the workspace with their metadata.
        
        Returns:
            Dictionary mapping audio IDs to their metadata summaries
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
