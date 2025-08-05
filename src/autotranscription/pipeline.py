"""
Auto-transcription pipeline for Indian classical music - Refactored Version.

Main orchestrator using modular components for better organization and maintainability.
"""

import time
import json
import shutil
from pathlib import Path
from typing import Union, Dict, Any, Optional

try:
    from .audio_separator import AudioSeparator
    from .pitch_extraction import PitchExtractor
    from .tonic_estimation import TonicEstimator
    from .visualization import SargamVisualizer
    from .audio_processing import AudioProcessor
    from .raga_estimation import RagaEstimator
except ImportError:
    from audio_separator import AudioSeparator
    from pitch_extraction import PitchExtractor
    from tonic_estimation import TonicEstimator
    from visualization import SargamVisualizer
    from audio_processing import AudioProcessor
    from raga_estimation import RagaEstimator


class AutoTranscriptionPipeline:
    """
    Complete pipeline for auto-transcription of Indian classical music.
    
    Orchestrates the following stages:
    1. Audio separation (vocals/instrumental)
    2. Tonic estimation
    3. Pitch trace extraction
    4. Raga estimation using TDMS
    5. Visualization generation
    6. Audio output creation
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
        self.visualizations_dir = self.workspace_dir / "visualizations"
        self.audio_outputs_dir = self.workspace_dir / "audio_outputs"
        
        for dir_path in [self.audio_dir, self.metadata_dir, self.separated_dir, 
                        self.data_dir, self.pitch_traces_dir, self.visualizations_dir,
                        self.audio_outputs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize modular components
        self.audio_separator = AudioSeparator(model_name=audio_separator_model)
        self.pitch_extractor = PitchExtractor(algorithms=algorithms)
        self.tonic_estimator = TonicEstimator()
        self.raga_estimator = RagaEstimator(models_dir=self.workspace_dir / "raga_models")
        self.visualizer = SargamVisualizer(self.visualizations_dir)
        self.audio_processor = AudioProcessor()
    
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
        
        # Stage 4: Raga Estimation
        if "raga_estimation" not in metadata["stages_completed"]:
            raga_results = self._run_raga_estimation(audio_id, metadata)
            metadata["results"]["raga_estimation"] = raga_results
            metadata["stages_completed"].append("raga_estimation")
            self.save_metadata(audio_id, metadata)
        else:
            raga_results = metadata["results"]["raga_estimation"]
        
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
        
        return raga_results
    
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