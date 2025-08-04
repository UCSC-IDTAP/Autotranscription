"""
Auto-transcription pipeline for Indian classical music.
"""

import time
import json
import shutil
from pathlib import Path
from typing import Union, Dict, Any, Optional
import essentia.standard as es

try:
    from .audio_separator import AudioSeparator
except ImportError:
    from audio_separator import AudioSeparator


class AutoTranscriptionPipeline:
    """
    Complete pipeline for auto-transcription of Indian classical music.
    """
    
    def __init__(self, audio_separator_model: str = "htdemucs", workspace_dir: Union[str, Path] = None):
        """
        Initialize the pipeline.
        
        Args:
            audio_separator_model: Demucs model to use for audio separation
            workspace_dir: Directory to store all pipeline files (optional)
        """
        self.audio_separator = AudioSeparator(model_name=audio_separator_model)
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "pipeline_workspace"
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.audio_dir = self.workspace_dir / "audio"
        self.metadata_dir = self.workspace_dir / "metadata"
        self.separated_dir = self.workspace_dir / "separated"
        
        for dir_path in [self.audio_dir, self.metadata_dir, self.separated_dir]:
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
        
        print(f"\n=== Auto-Transcription Pipeline ===")
        print(f"Audio ID: {audio_id}")
        print(f"Original file: {metadata['original_name']}")
        print(f"Workspace directory: {self.workspace_dir}")
        
        pipeline_start = time.time()
        
        # Step 1: Audio Separation
        if "audio_separation" not in metadata["stages_completed"]:
            print(f"\n--- Step 1: Audio Separation ---")
            separation_results = self._run_audio_separation(audio_id, metadata)
            metadata["results"]["audio_separation"] = separation_results
            metadata["stages_completed"].append("audio_separation")
            self.save_metadata(audio_id, metadata)
        else:
            print(f"\n--- Step 1: Audio Separation (SKIPPED - already completed) ---")
            separation_results = metadata["results"]["audio_separation"]
        
        # Step 2: Tonic Estimation
        if "tonic_estimation" not in metadata["stages_completed"]:
            print(f"\n--- Step 2: Tonic Estimation ---")
            tonic_results = self._run_tonic_estimation(audio_id, metadata)
            metadata["results"]["tonic_estimation"] = tonic_results
            metadata["stages_completed"].append("tonic_estimation")
            self.save_metadata(audio_id, metadata)
        else:
            print(f"\n--- Step 2: Tonic Estimation (SKIPPED - already completed) ---")
            tonic_results = metadata["results"]["tonic_estimation"]
        
        pipeline_time = time.time() - pipeline_start
        metadata["last_run_time"] = pipeline_time
        self.save_metadata(audio_id, metadata)
        
        print(f"\n=== Pipeline Complete ===")
        print(f"Total time: {pipeline_time:.2f} seconds")
        print(f"Metadata saved: {self.metadata_dir / f'{audio_id}.json'}")
        
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
        
        print(f"Audio separation completed in {results['separation_time']:.2f} seconds")
        print(f"Vocals: {results['vocals']}")
        print(f"Instrumental: {results['instrumental']}")
        
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
        print(f"Original track tonic: {original_tonic:.2f} Hz ({results['original']['tonic_note']})")
        
        # Test on separated instrumental track
        print("Estimating tonic on separated instrumental track...")
        instrumental_tonic = self._estimate_tonic(instrumental_path)
        results['instrumental'] = {
            "file": str(instrumental_path),
            "tonic_frequency": instrumental_tonic,
            "tonic_note": self._frequency_to_note(instrumental_tonic)
        }
        print(f"Instrumental track tonic: {instrumental_tonic:.2f} Hz ({results['instrumental']['tonic_note']})")
        
        # Compare results
        frequency_diff = abs(original_tonic - instrumental_tonic)
        cents_diff = 1200 * abs(original_tonic - instrumental_tonic) / min(original_tonic, instrumental_tonic) if min(original_tonic, instrumental_tonic) > 0 else 0
        
        results['comparison'] = {
            "frequency_difference": frequency_diff,
            "cents_difference": cents_diff
        }
        
        print(f"Frequency difference: {frequency_diff:.2f} Hz ({cents_diff:.1f} cents)")
        
        return results
    
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
        
        print(f"\n=== Final Results ===")
        print(f"Audio ID: {results['id']}")
        print(f"Pipeline completed in {results.get('last_run_time', 0):.2f} seconds")
        print(f"Workspace: {workspace_dir}")
        print(f"Stages completed: {', '.join(results['stages_completed'])}")
        
        # Show tonic estimation results if available
        if 'tonic_estimation' in results.get('results', {}):
            tonic_results = results['results']['tonic_estimation']
            print(f"\nTonic Estimation Results:")
            print(f"  Original: {tonic_results['original']['tonic_frequency']:.2f} Hz ({tonic_results['original']['tonic_note']})")
            print(f"  Instrumental: {tonic_results['instrumental']['tonic_frequency']:.2f} Hz ({tonic_results['instrumental']['tonic_note']})")
            print(f"  Difference: {tonic_results['comparison']['frequency_difference']:.2f} Hz ({tonic_results['comparison']['cents_difference']:.1f} cents)")
        
    else:
        print(f"Audio file not found: {audio_file}")
        print(f"Please ensure the audio file exists at: {audio_file}")
        print(f"Project root detected as: {project_root}")
        
        # Show existing audio files in workspace if any
        pipeline = AutoTranscriptionPipeline(workspace_dir=workspace_dir)
        existing_files = pipeline.list_audio_files()
        if existing_files:
            print(f"\nExisting audio files in workspace:")
            for audio_id, info in existing_files.items():
                print(f"  {audio_id}: {info['original_name']} - stages: {', '.join(info['stages_completed'])}")