"""
Audio separation module using Demucs for vocal/instrumental separation.
"""

import os
import time
from pathlib import Path
from typing import Union, List, Dict
import demucs.separate


class AudioSeparator:
    """
    A class for separating vocals from instrumental audio using Demucs.
    """
    
    def __init__(self, model_name: str = "htdemucs"):
        """
        Initialize the AudioSeparator.
        
        Args:
            model_name: The Demucs model to use for separation
        """
        self.model_name = model_name
    
    def separate_vocals(self, 
                       input_path: Union[str, Path], 
                       output_dir: Union[str, Path] = None) -> dict:
        """
        Separate vocals from the rest of the audio.
        
        Args:
            input_path: Path to the input audio file
            output_dir: Directory to save separated tracks (optional)
            
        Returns:
            dict: Paths to separated vocal and instrumental tracks
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Set up output directory
        if output_dir is None:
            output_dir = input_path.parent / "separated"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Prepare arguments for demucs
        args = [
            "--two-stems", "vocals",  # Only separate vocals vs rest
            "-n", self.model_name,    # Model name
            "-o", str(output_dir),    # Output directory
            str(input_path)           # Input file
        ]
        
        print(f"Separating audio: {input_path.name}")
        print(f"Using model: {self.model_name}")
        print(f"Output directory: {output_dir}")
        
        # Run demucs separation with timing
        start_time = time.time()
        demucs.separate.main(args)
        end_time = time.time()
        separation_time = end_time - start_time
        
        print(f"Separation completed in {separation_time:.2f} seconds")
        
        # Construct expected output paths
        stem_name = input_path.stem
        model_output_dir = output_dir / self.model_name / stem_name
        
        vocal_path = model_output_dir / "vocals.wav"
        instrumental_path = model_output_dir / "no_vocals.wav"
        
        return {
            "vocals": vocal_path,
            "instrumental": instrumental_path,
            "output_dir": model_output_dir,
            "separation_time": separation_time,
            "model": self.model_name
        }


def separate_audio_file(input_file: Union[str, Path], 
                       output_dir: Union[str, Path] = None,
                       model: str = "htdemucs") -> dict:
    """
    Convenience function to separate vocals from an audio file.
    
    Args:
        input_file: Path to the input audio file
        output_dir: Directory to save separated tracks (optional)
        model: Demucs model to use
        
    Returns:
        dict: Paths to separated tracks
    """
    separator = AudioSeparator(model_name=model)
    return separator.separate_vocals(input_file, output_dir)


def compare_models(input_file: Union[str, Path], 
                  models: List[str] = None,
                  output_dir: Union[str, Path] = None) -> Dict[str, dict]:
    """
    Compare multiple Demucs models on the same audio file.
    
    Args:
        input_file: Path to the input audio file
        models: List of model names to compare
        output_dir: Directory to save separated tracks (optional)
        
    Returns:
        dict: Results for each model with timing information
    """
    if models is None:
        models = ["htdemucs", "htdemucs_ft"]
    
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    results = {}
    total_start = time.time()
    
    print(f"\n=== Comparing {len(models)} models on {input_path.name} ===")
    print(f"Models to test: {', '.join(models)}\n")
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Testing model: {model}")
        print("-" * 50)
        
        try:
            result = separate_audio_file(input_file, output_dir, model)
            results[model] = result
            
            # Check if files were created
            vocals_exists = result['vocals'].exists()
            instrumental_exists = result['instrumental'].exists()
            
            print(f"✓ Vocals saved: {vocals_exists}")
            print(f"✓ Instrumental saved: {instrumental_exists}")
            
        except Exception as e:
            print(f"✗ Error with {model}: {e}")
            results[model] = {"error": str(e)}
    
    total_time = time.time() - total_start
    
    print(f"\n=== Comparison Summary ===")
    print(f"Total time: {total_time:.2f} seconds\n")
    
    for model, result in results.items():
        if "error" in result:
            print(f"{model:12}: ERROR - {result['error']}")
        else:
            print(f"{model:12}: {result['separation_time']:.2f}s")
    
    # Calculate speed ratios
    if len([r for r in results.values() if "error" not in r]) >= 2:
        times = {model: result['separation_time'] 
                for model, result in results.items() 
                if "error" not in result}
        
        if "htdemucs" in times and "htdemucs_ft" in times:
            speed_ratio = times["htdemucs_ft"] / times["htdemucs"]
            print(f"\nSpeed ratio (htdemucs_ft / htdemucs): {speed_ratio:.1f}x slower")
    
    return results


if __name__ == "__main__":
    # Model comparison test
    project_root = Path(__file__).parent.parent.parent
    audio_file = project_root / "audio" / "Kishori Amonkar - Babul Mora_1min.mp3"
    
    if audio_file.exists():
        # Compare htdemucs and htdemucs_ft
        results = compare_models(audio_file, ["htdemucs", "htdemucs_ft"])
        
        print(f"\n=== Results saved to ===")
        for model, result in results.items():
            if "error" not in result:
                print(f"{model}: {result['output_dir']}")
                
    else:
        print(f"Audio file not found: {audio_file}")
        print(f"Please ensure the audio file exists at: {audio_file}")
        print(f"Project root detected as: {project_root}")
