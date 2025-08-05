# Auto-Transcription Pipeline for Indian Classical Music

## Project Overview
This project implements an automatic transcription pipeline specifically designed for Indian classical music. The pipeline processes audio files through multiple stages to extract pitch information and generate visual/audio representations of the melodic content.

## Architecture

### Core Components

1. **AudioSeparator** (`src/autotranscription/audio_separator.py`)
   - Uses Demucs models (htdemucs, htdemucs_ft) for vocal/instrumental separation
   - Separates input audio into vocals and instrumental tracks
   - Creates organized output structure with timing information

2. **AutoTranscriptionPipeline** (`src/autotranscription/pipeline.py`)
   - Main orchestrator for the entire transcription process
   - Manages workspace organization and file tracking
   - Implements resumable pipeline with stage completion tracking

### Pipeline Stages

#### Stage 1: Audio Separation
- Uses Demucs to separate vocals from instrumental tracks
- Creates `vocals.wav` and `no_vocals.wav` files
- Stores separation metadata and timing

#### Stage 2: Tonic Estimation
- Uses Essentia's `TonicIndianArtMusic` algorithm
- Estimates tonic frequency on both original and instrumental tracks
- Compares results and calculates frequency/cents differences

#### Stage 3: Pitch Trace Extraction
- **Primary Algorithm**: Essentia's `PredominantPitchMelodia` (optimized for quiet sections)
- **Optional Algorithms**: CREPE, SWIPE, PyWorld
- Parameters tuned for Indian classical music (55-1760 Hz range)
- Confidence-based filtering with algorithm-specific thresholds

### Key Features

#### Workspace Management
- Organized directory structure under `pipeline_workspace/`
- Unique numeric IDs for each audio file
- JSON metadata tracking for resumable processing
- Stage completion tracking to avoid reprocessing

#### Directory Structure
```
pipeline_workspace/
├── audio/           # Copied input files
├── metadata/        # JSON files with processing metadata
├── separated/       # Demucs output (vocals/instrumental)
├── data/
│   └── pitch_traces/  # Numpy arrays with pitch data
├── visualizations/  # PNG plots comparing algorithms
└── audio_outputs/   # Sine wave reconstructions
```

#### Pitch Extraction Optimization
- **Essentia**: Optimized for quiet sections with `voicingTolerance=0.4`, `guessUnvoiced=True`, `magnitudeThreshold=20`
- **Confidence Thresholds**: Essentia (0.05), CREPE (0.7), SWIPE (0.5), PyWorld (0.5)
- **Frequency Range**: 55-1760 Hz (appropriate for Indian classical vocals)

#### Visualization & Audio Output
- Comparative pitch trace plots with logarithmic frequency scale
- Individual algorithm plots showing confidence-filtered results
- Sine wave reconstructions with original amplitude envelope applied
- Disconnected segments for low-confidence regions (no interpolation)

## Usage Patterns

### Basic Usage
```python
from autotranscription.pipeline import run_pipeline

# Process a single file
results = run_pipeline("audio_file.mp3", workspace_dir="my_workspace")

# Resume processing (automatically detects existing work)
results = run_pipeline("audio_file.mp3", workspace_dir="my_workspace")
```

### Advanced Usage
```python
# Use specific algorithms
pipeline = AutoTranscriptionPipeline(
    workspace_dir="workspace",
    algorithms=['essentia', 'crepe']  # Default: ['essentia']
)
results = pipeline.run("audio_file.mp3")

# List processed files
audio_files = pipeline.list_audio_files()
```

## Dependencies
- **Audio Processing**: demucs, essentia, librosa, soundfile
- **Pitch Estimation**: crepe, pysptk (SWIPE), pyworld
- **Scientific**: numpy, scipy, matplotlib
- **Environment**: pipenv for dependency management

## Current State
- ✅ Complete pipeline with audio separation, tonic estimation, and pitch extraction
- ✅ Optimized for Indian classical music characteristics
- ✅ Resumable processing with metadata tracking
- ✅ Multiple pitch estimation algorithms with confidence filtering
- ✅ Visualization and audio output generation
- ✅ Workspace organization and file management

## Testing
- Uses 1-minute excerpt of "Kishori Amonkar - Babul Mora" for testing
- Pipeline creates comprehensive outputs in `pipeline_workspace/`
- Visualizations and sine wave reconstructions available for quality assessment

## Notes for Future Development
- Default configuration uses only Essentia for efficiency
- Other algorithms (CREPE, SWIPE, PyWorld) available for comparison
- Confidence thresholds tuned for Indian classical music
- Amplitude envelope preservation in sine wave generation
- Logarithmic frequency visualization for musical analysis

## Commands to Remember
- **Test Pipeline**: `pipenv run python src/autotranscription/pipeline.py`
- **Test Audio Separator**: `pipenv run python src/autotranscription/audio_separator.py`
- **Environment Setup**: `pipenv install` then `pipenv shell`