"""
Audio separation module using Demucs for vocal/instrumental separation.
"""

import os
from pathlib import Path
from typing import Union
import demucs.separate


class AudioSeparator:
    """
    A class for separating vocals from instrumental audio using Demucs.
    """
    
    def __init__(self, model_name: str = "mdx_extra"):
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
        
        # Run demucs separation
        demucs.separate.main(args)
        
        # Construct expected output paths
        stem_name = input_path.stem
        model_output_dir = output_dir / self.model_name / stem_name
        
        vocal_path = model_output_dir / "vocals.wav"
        instrumental_path = model_output_dir / "no_vocals.wav"
        
        return {
            "vocals": vocal_path,
            "instrumental": instrumental_path,
            "output_dir": model_output_dir
        }


def separate_audio_file(input_file: Union[str, Path], 
                       output_dir: Union[str, Path] = None,
                       model: str = "mdx_extra") -> dict:
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


if __name__ == "__main__":
    # Example usage - find project root and audio file
    # Get the project root (two levels up from this script)
    project_root = Path(__file__).parent.parent.parent
    audio_file = project_root / "audio" / "Babul_mora.wav"
    
    if audio_file.exists():
        print(f"Separating vocals from {audio_file.name}...")
        result = separate_audio_file(audio_file)
        print("\nSeparation complete!")
        print(f"Vocals: {result['vocals']}")
        print(f"Instrumental: {result['instrumental']}")
    else:
        print(f"Audio file not found: {audio_file}")
        print(f"Please ensure the audio file exists at: {audio_file}")
        print(f"Project root detected as: {project_root}")
